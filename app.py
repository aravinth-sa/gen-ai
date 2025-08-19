## Integration with OpenAI API

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DataFrameLoader
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.schema.runnable import RunnableMap
import logging

# Set the API key
# Load API Key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
#openai_api_key = st.secrets["OPENAI_API_KEY"]

############################################
# Load data first using PineCone
############################################
import pinecone
from langchain.vectorstores import Pinecone

@st.cache_resource
def load_and_embed_pinecone():
    # Load data
    df = pd.read_csv("dataset100.csv")
    df["text"] = df.apply(
        lambda row: f"Product Code: {row['pid']}\nProduct: {row['product_name']}\nDescription: {row['description']}\nCategory: {row['product_category_tree']}\nPrice: ${row['retail_price']}",
        axis=1
    )
    docs = DataFrameLoader(df[["text"]], page_content_column="text").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Pinecone setup
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    index_name = "product-index"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)  # 1536 for OpenAI embeddings

    vectorstore = Pinecone.from_documents(split_docs, embeddings, index_name=index_name)
    return vectorstore
############################################
# Load data first using FAISS
############################################
@st.cache_resource
def load_and_embed():
    df = pd.read_csv("dataset100.csv")
    df.head()
    df["text"] = df.apply(
        lambda row: f"Product Code: {row['pid']}\nProduct: {row['product_name']}\nDescription: {row['description']}\nCategory: {row['product_category_tree']}\nPrice: ${row['retail_price']}",
        axis=1
    )
    docs = DataFrameLoader(df[["text"]], page_content_column="text").load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

vectorstore = load_and_embed()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# A custom prompt made for chain
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
                You are a helpful e-commerce assistant for a building materials and home furnishing store.
                Use the product information below to answer the customer's question.

                Product Info:
                {context}

                Customer Question:
                {question}

                Helpful Answer (based only on the product info above):
                """
                )

# Setup RetrievalQA
def get_qa_chain(retriever):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )
    return chain





########### FUNCTIONS ############

# Define a function to get product price
def get_product_price(product_code: str) -> str:
    df = pd.read_csv("dataset100.csv")
    product = df[
        (df["pid"] == product_code) |
        (df["product_name"].astype(str).str.contains(product_code, case=False, na=False)) |
        (df["description"].astype(str).str.contains(product_code, case=False, na=False))
    ]
    if not product.empty:
        price = product.iloc[0]["retail_price"]
        name = product.iloc[0]["product_name"]
        return f"The price of {name} (Product Code: {product_code}) is ${price}."
    else:
        return "Product not found."

# Define a function to get product description
def get_product_description(product_code: str) -> str:
    df = pd.read_csv("dataset100.csv")
    product = df[
        (df["pid"] == product_code) |
        (df["product_name"].astype(str).str.contains(product_code, case=False, na=False)) |
        (df["description"].astype(str).str.contains(product_code, case=False, na=False))
    ]
    if not product.empty:
        desc = product.iloc[0]["description"]
        name = product.iloc[0]["product_name"]
        return f"Description for {name} (Product Code: {product_code}): {desc}"
    else:
        return "Product not found."
    
def qa_tool_func(question):
    logging.info(f"qa_tool_func called with question: {question}")
    df = pd.read_csv("dataset100.csv")
    # Check if question contains a known product code
    for pid in df["pid"].astype(str):
        if pid in question:
            logging.info(f"Matched product code: {pid}")
            context = df[df["pid"].astype(str) == pid]["text"].values
            if len(context) > 0:
                logging.info("Returning context by product code match.")
                return context[0]
    # Otherwise, try to match product name (case-insensitive, partial match)
    question_lower = question.lower()
    for name in df["product_name"].astype(str):
        name_lower = name.lower()
        if name_lower in question_lower or question_lower in name_lower:
            logging.info(f"Matched product name: {name}")
            context = df[df["product_name"].astype(str).str.lower() == name_lower]["text"].values
            if len(context) > 0:
                logging.info("Returning context by product name match.")
                return context[0]
    # Fallback to vector search
    logging.info("No product code or name match found. Using vector search.")
    result = qa_chain({"query": question})["result"]
    logging.info("Returning result from vector search.")
    return result

########### TOOLS ###########

# Bind functions with LCEL (LangChain Expression Language)
price_tool = Tool.from_function(
    func=get_product_price,
    name="get_product_price",
    description="Get the price of a product by product code."
)

desc_tool = Tool.from_function(
    func=get_product_description,
    name="get_product_description",
    description="Get the description of a product by product code."
)

qa_chain = get_qa_chain(retriever)

qa_tool = Tool.from_function(
    func=qa_tool_func,
    name="product_qa",
    description="Answer questions about products using embedded product information, product code, or product name."
)

# LCEL chain composition
#chain = RunnableMap({
#    "qa": qa_chain,
#    "price": price_tool,
#    "description": desc_tool
#})

# Agent setup using tools and chain
#tools = [price_tool, desc_tool]
# The previous agent setup only uses the tools (price/description) and does NOT use the RetrievalQA chain (which leverages the vectorstore embeddings).
# To ensure the agent uses the embedded vector information, you should add a tool that wraps the qa_chain.

tools = [price_tool, desc_tool, qa_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

def display_chat_history():
    # Initialize messages in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display previous messages in the chat
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")

def get_user_input():
    # Get user input from the text box
    return st.text_input("Your message", key="chat_input", placeholder="Type your question here...")

def process_user_input(user_input):
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize last_user_input in session state if not present
    if "last_user_input" not in st.session_state:
        st.session_state.last_user_input = ""

    # If there is new user input, process it
    if user_input and user_input != st.session_state.last_user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        logging.info(f"User input: {user_input}")

        try:
            # Get response from the agent
            response = agent({"input": user_input})
            logging.info(f"Agent raw response: {response}")
            answer = response.get("output", "No output from agent.")
        except Exception as e:
            logging.error(f"Error during agent response: {e}")
            answer = f"Sorry, there was an error processing your request: {e}"

        # Append assistant's answer to the chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Update last_user_input and rerun the app to refresh UI
        st.session_state.last_user_input = user_input
        st.rerun()

def handle_chat_interaction():
    display_chat_history()
    user_input = get_user_input()
    process_user_input(user_input)

# Call the chat interaction handler
handle_chat_interaction()

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


