import os
import chromadb
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")  # ⚡ memory-efficient

def get_answer(query: str) -> str:
    try:
        model = load_model()
        chroma = chromadb.PersistentClient(path="chroma_store")
        collection = chroma.get_collection("legal_docs")

        embedding = model.encode([query])[0].tolist()
        results = collection.query(query_embeddings=[embedding], n_results=3)
        top_docs = results["documents"][0]

        if not any(len(doc.strip()) > 30 for doc in top_docs):
            return "❌ Sorry, no relevant section was found in the legal database."

        context = "\n\n".join(top_docs)

        prompt = f"""Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
        return call_groq(prompt)

    except Exception as e:
        return f"❌ Internal Error: {str(e)}"

def call_groq(prompt: str) -> str:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        return "❌ GROQ API Key not found in environment."
    

    print("Running Groq")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    

    body = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "system",
                "content": "You are a legal assistant. Answer using only the context. If unsure, say so clearly."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body)
 
    if res.status_code != 200:
        return f"⚠️ Groq API Error {res.status_code}: {res.text}"

    print(res.json()["choices"][0]["message"]["content"])
    return res.json()["choices"][0]["message"]["content"]
