import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from semantic_search import semantic_search
from graph_search import get_kural_details

load_dotenv()


# def thirukkural_chat_kg(user_query, metadata):
#     """
#     Using LangChain with Groq for Thirukkural queries
#     """
#     api_key = os.getenv("GROQ_API_KEY")
#     # Initialize ChatGroq
#     llm = ChatGroq(
#         api_key=api_key,
#         model_name="llama-3.1-70b-versatile",
#         temperature=0.3
#     )
    
#     # Create prompt template
#     system_prompt = """You are an expert on Thirukkural, the ancient Tamil ethical text by Thiruvalluvar.

# IMPORTANT RULES:
# 1. If the user's question is related to Thirukkural, Tamil literature, ethics, wisdom, or moral teachings, answer using the provided metadata.
# 2. If the question is NOT related to Thirukkural, respond ONLY with: "This query is not related to Thirukkural. Please ask about Thirukkural verses, their meanings, or related topics."

# Available Thirukkural Data:
# {metadata}

# Answer based only on the provided Thirukkural context above."""

#     human_prompt = "User Question: {user_query}"
    
#     # Create prompt template
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", human_prompt)
#     ])
    
#     # Format metadata as string
#     metadata_str = json.dumps(metadata, indent=2, ensure_ascii=False)
    
#     # Create chain and invoke
#     chain = prompt | llm
#     response = chain.invoke({
#         "metadata": metadata_str,
#         "question": user_query
#     })
    
#     return response.content
def thirukkural_chat_langchain(user_query, metadata):
    """
    Using LangChain with Groq for Thirukkural queries
    """
    api_key = os.getenv("GROQ_API_KEY")
    # Initialize ChatGroq
    llm = ChatGroq(
        api_key=api_key,
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0.3
    )
    
    # Create prompt template
    system_prompt = """You are an expert on Thirukkural, the ancient Tamil ethical text by Thiruvalluvar.

IMPORTANT RULES:
1. If the user's question is related to Thirukkural, Tamil literature, ethics, wisdom, or moral teachings, answer using the provided metadata.
2. If the question is NOT related to Thirukkural, respond ONLY with: "This query is not related to Thirukkural. Please ask about Thirukkural verses, their meanings, or related topics."
3. You must go for all the available metadata, think of the query and give a meaningful text with context to all the given. Make sure to select two thirukkural that are most relevent for the input query, 
    and make it as an example and explain to the user.
4. Make sure to response in both the language Tamil and English. 
5. If a number from 1 to 1330 is given.. kindly give about the kural that is specified. 
Available Thirukkural Data:
{metadata}

Answer based only on the provided Thirukkural context above or relatable kural."""

    human_prompt = "User Question: {question}"
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    # Format metadata as string
    metadata_str = json.dumps(metadata, indent=2, ensure_ascii=False)
    
    # Create chain and invoke - FIXED: Use 'question' not 'user_query'
    chain = prompt | llm
    response = chain.invoke({
        "metadata": metadata_str,
        "question": user_query  # Changed from "question" to match template variable
    })
    
    return response.content

# Flow
query = input("Enter for the query:")
res = semantic_search(query)
r = []
for i in res:
    r.append(i['number'])
response = get_kural_details(r)
# print(response)
print(thirukkural_chat_langchain(query, response))