import streamlit as st
from langchain.memory import ConversationBufferMemory
from data import load_data
from embedding import get_faiss_vectorstore
from agent import get_qa_chain, get_agent
from tools import get_tools
from ui import handle_chat_interaction

df = load_data()
vectorstore = get_faiss_vectorstore(df)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = get_qa_chain(retriever)
tools = get_tools(qa_chain)
agent = get_agent(tools, memory)

handle_chat_interaction(agent)


