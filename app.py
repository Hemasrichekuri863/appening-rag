from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from agent import get_agent
app = FastAPI(title="Appening AI RAG")
agent = get_agent()
class Query(BaseModel):
    question: str
@app.get("/")
def home():
    return {"message": "Appening AI RAG is running!"}
@app.post("/chat")
def chat(q: Query):
    result = agent.invoke({"messages": [("user", q.question)]})
    last_msg = result["messages"][-1].content
    return {"answer": last_msg}
