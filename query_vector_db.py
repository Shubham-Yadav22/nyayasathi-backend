import os
import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global variables for lazy loading
model = None
collection = None

def get_model():
    """Lazy load the sentence transformer model"""
    global model
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence transformer model...")
            # Use the smallest, fastest model
            model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return model

def get_collection():
    """Lazy load the ChromaDB collection"""
    global collection
    if collection is None:
        try:
            import chromadb
            logger.info("Connecting to ChromaDB...")
            
            # Setup ChromaDB from local persistent directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            persist_dir = os.path.join(script_dir, "chroma_store")
            
            if not os.path.exists(persist_dir):
                logger.error(f"ChromaDB directory not found: {persist_dir}")
                raise FileNotFoundError(f"Database not found at {persist_dir}")
            
            chroma = chromadb.PersistentClient(path=persist_dir)
            collection = chroma.get_collection("legal_docs")
            logger.info("ChromaDB connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    return collection

def answer_query(query: str) -> str:
    """Process a legal query and return an answer"""
    
    if not query or len(query.strip()) < 3:
        return "❌ Please provide a valid query with at least 3 characters."
    
    try:
        logger.info("📥 Processing query...")
        
        # Get model and collection (lazy loaded)
        model = get_model()
        collection = get_collection()
        
        # Generate query embedding
        logger.info("🔍 Generating query embedding...")
        query_embedding = model.encode([query.strip()])[0].tolist()
        
        # Retrieve top relevant documents
        logger.info("📚 Searching relevant documents...")
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=3,  # Increased slightly for better context
            include=["documents", "metadatas"]
        )
        
        top_docs = results.get("documents", [[]])[0]
        
        # Filter out irrelevant/empty results
        relevant_docs = [doc.strip() for doc in top_docs if len(doc.strip()) > 50]
        
        if not relevant_docs:
            return "❌ No relevant legal information found for your query. Please try rephrasing your question."
        
        # Build context prompt (limit context size)
        context = "\n\n".join(relevant_docs[:2])  # Limit to top 2 docs
        if len(context) > 3000:  # Truncate if too long
            context = context[:3000] + "..."
            
        return call_groq_api(query, context)
        
    except FileNotFoundError as e:
        logger.error(f"Database error: {e}")
        return "❌ Legal database is not available. Please contact support."
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"❌ An error occurred while processing your query. Please try again."

def call_groq_api(query: str, context: str) -> str:
    """Call Groq API with the query and context"""
    
    # Load Groq API Key from environment
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not found in environment")
        return "❌ API configuration error. Please contact support."
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Optimized prompt for better responses
    system_prompt = (
        "You are a legal assistant for Indian law. Use ONLY the provided legal context to answer questions. "
        "Provide clear, concise answers in simple language. If the context doesn't contain enough information "
        "to answer the question, clearly state that. Always cite relevant sections or acts when available."
    )
    
    user_prompt = f"Legal Context:\n{context}\n\nUser Question: {query}\n\nProvide a helpful legal answer based on the context above."
    
    body = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 500,  # Limit response length
        "temperature": 0.1  # More consistent responses
    }
    
    try:
        logger.info("📡 Querying Groq API...")
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions", 
            headers=headers, 
            json=body,
            timeout=30  # Add timeout
        )
        
        if response.status_code != 200:
            logger.error(f"Groq API error: {response.status_code} - {response.text}")
            return "❌ Unable to process your query at the moment. Please try again later."
        
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        
        logger.info("✅ Successfully generated response")
        return answer.strip()
        
    except requests.exceptions.Timeout:
        logger.error("Groq API timeout")
        return "❌ Request timed out. Please try again."
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return "❌ Network error occurred. Please check your connection and try again."
        
    except Exception as e:
        logger.error(f"Unexpected error in Groq API call: {e}")
        return "❌ An unexpected error occurred. Please try again."