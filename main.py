import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query_handler import get_answer


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from query_handler import get_answer

app = FastAPI()

# Allow frontend access (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://nyaya-sathi-phi.vercel.app"
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
async def handle_query(req: Request):
    data = await req.json()
    query = data.get("message", "")
    
    if not query.strip():
        return {"reply": "⚠️ Please enter a valid legal question."}

    reply = get_answer(query)
    return {"reply": reply}
