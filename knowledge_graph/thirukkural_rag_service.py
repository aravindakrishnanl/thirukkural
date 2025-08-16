from fastapi import FastAPI
from pydantic import BaseModel
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import uvicorn

print("Loading dataset: Selvakumarduraipandian/Thirukural")
dataset = load_dataset("Selvakumarduraipandian/Thirukural", split="train")
kurals = dataset


class BaseRetriever:
    def __init__(self, kurals, field='Couplet'):
        self.kurals = kurals
        self.field = field

    def retrieve(self, query, top_k=5):
        raise NotImplementedError

    def format_result(self, res):
        return {
            "ID": res['ID'],
            "Kural_Tamil": res['Kural'],
            "Kural_English": res['Couplet'],
            "Vilakam": res['Vilakam']
        }


class KeywordRetriever(BaseRetriever):
    def retrieve(self, query, top_k=5):
        query_words = set(re.findall(r'\w+', query.lower()))
        scores = []
        for kural in self.kurals:
            text = kural[self.field].lower()
            match_count = sum(1 for word in query_words if word in text)
            scores.append(match_count)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.kurals[int(i)] for i in top_indices if scores[i] > 0]


class TFIDFRetriever(BaseRetriever):
    def __init__(self, kurals, field='Couplet'):
        super().__init__(kurals, field)
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform([kural[self.field] for kural in kurals])

    def retrieve(self, query, top_k=5):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.kurals[int(i)] for i in top_indices]


class EmbeddingRetriever(BaseRetriever):
    def __init__(self, kurals, field='Couplet'):
        super().__init__(kurals, field)
        print("Loading embedding model: all-MiniLM-L6-v2")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode([kural[self.field] for kural in kurals])

    def retrieve(self, query, top_k=5):
        query_emb = self.model.encode([query])
        similarities = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.kurals[int(i)] for i in top_indices]


retrievers = {
    'Keyword': KeywordRetriever(kurals),
    'TF-IDF': TFIDFRetriever(kurals),
    'Embedding': EmbeddingRetriever(kurals),
}


app = FastAPI(title="Thirukkural Multi-Retriever API")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/ask")
def ask_kural(req: QueryRequest):
    results = {}
    for name, retriever in retrievers.items():
        retrieved_kurals = retriever.retrieve(req.query, req.top_k)
        results[name] = [retriever.format_result(res) for res in retrieved_kurals]
    return {
        "query": req.query,
        "results": results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
