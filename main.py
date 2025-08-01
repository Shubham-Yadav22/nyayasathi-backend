import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Nyaya Sathi API", version="1.0.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://nyaya-sathi-three.vercel.app"
    ], 
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    message: str  # Changed from 'query' to 'message' to match your frontend

@app.get("/")
async def root():
    return {"status": "Nyaya Sathi Backend is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/query")
async def query_handler(data: QueryRequest):
    try:
        logger.info(f"Received query: {data.message}")
        
        # Lazy import to reduce startup time
        from query_vector_db import answer_query
        
        result = answer_query(data.message)
        logger.info("Query processed successfully")
        
        return {"reply": result}  # Changed from 'response' to 'reply' to match your frontend
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)