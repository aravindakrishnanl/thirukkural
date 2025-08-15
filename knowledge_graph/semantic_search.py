import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

embeddings = np.load("thirukkural_embs.npy")
with open("thirukkural_rows.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Cosine Similarity Search
embeddings = normalize(embeddings, axis=1).astype("float32")

# Faiss Index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
# print(f"✅ FAISS index ready with {len(embeddings)} vectors")

# Load the SentenceTransformer model once
# print(f"📦 Loading embedding model: {MODEL_NAME} ({DEVICE})")
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_search(query: str, top_k: int = 3):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    q_emb = q_emb.astype("float32")
    
    D, I = index.search(q_emb, top_k)
    
    results = []
    for idx, score in zip(I[0], D[0]):
        results.append({
            "score": float(score),
            **metadata[idx]
        })
    return results


# query = "virtue and kindness"
# hits = semantic_search(query, top_k=5)
# # Attributes: score, number, kural, translation, section, chapter
# print(hits)
# for i, hit in enumerate(hits, 1):
#     print(f"{i}. Score={hit['score']:.4f} | ID={hit['number']}")
#     print(f"   Kural (TA): {hit['kural']}")
#     # print(f"   Kural (EN): {hit['Couplet']}")
#     print()
