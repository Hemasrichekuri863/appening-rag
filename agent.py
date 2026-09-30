import os
from langchain_groq import ChatGroq
from src.tools import get_retriever
def get_agent():
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"))
    retriever = get_retriever()
    def rag_chain(inputs):
        q = inputs["question"]
        docs = retriever.invoke(q)
        print(f"RETRIEVED {len(docs)} docs")
        if not docs:
            return {"answer": f"Index {os.getenv('PINECONE_INDEX_NAME')} returned 0 docs for '{q}'. Ingestion may not have reached Pinecone yet."}
        ctx = "\n\n".join([d.page_content for d in docs[:4]])
        resp = llm.invoke(f"Context:{ctx}\n\nQ:{q}\nAnswer briefly:")
        return {"answer": resp.content}
    return type("Agent", (), {"invoke": rag_chain})()
