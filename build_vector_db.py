import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# Load your BNS legal data
DATA_PATH = "data/bns.json"  # Adjust path if needed

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ File not found: {DATA_PATH}")

# Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize model and ChromaDB
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
persist_dir = os.path.join(os.getcwd(), "chroma_store")
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_or_create_collection(name="legal_docs")

# Prepare documents and IDs
documents = []
ids = []

for i, item in enumerate(data):
    section = item.get("bns_section", "")
    title = item.get("subject", "")
    content = item.get("extra_data", "")
    summary= item.get("summary", "")
    text = f"BNS Section: {section}\nSubject: {title}\n\n{content}\n\n{summary}"
    documents.append(text)
    ids.append(f"doc_{i}")

# Embed & store
print("🔍 Generating embeddings...")
embeddings = model.encode(documents).tolist()

print("💾 Storing vectors in Chroma...")
collection.add(documents=documents, embeddings=embeddings, ids=ids)

print(f"✅ ChromaDB vector store built and saved to: {persist_dir}")
