import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Appening RAG")

class Question(BaseModel):
    question: str

_agent = None

def get_agent_lazy():
    global _agent
    if _agent is None:
        # import here, not at top - prevents 502 on startup
        try:
            from src.agent import get_agent
        except:
            from agent import get_agent
        _agent = get_agent()
    return _agent

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG API is live"}

@app.post("/chat")
def chat(q: Question):
    try:
        agent = get_agent_lazy()
        result = agent.invoke({"question": q.question})
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "hint": "Check Pinecone has 119 vectors and GROQ key valid"}
