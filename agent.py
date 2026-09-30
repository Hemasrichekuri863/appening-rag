from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from src.tools import retrieve_context

def get_agent():
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    tools = [retrieve_context]
    agent = create_react_agent(llm, tools)
    return agent