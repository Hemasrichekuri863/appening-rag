from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI(title="Appening RAG")

# Lazy load - don't load at startup!
agent = None

def get_agent_lazy():
    global agent
    if agent is None:
        from agent import get_agent
        agent = get_agent()
    return agent

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG API Live"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        ag = get_agent_lazy()
        result = ag.invoke({"question": req.question})
        # handle different return formats
        if isinstance(result, dict):
            answer = result.get("answer") or result.get("output") or str(result)
        else:
            answer = str(result)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e), "hint": "Check Pinecone index has vectors and API keys are correct"}
