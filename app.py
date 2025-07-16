## Integration with OpenAI API

import os
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DataFrameLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st

# Set the API key
# Load API Key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load data
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




if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about the product catalog."}
    ]

vectorstore = load_and_embed()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = get_qa_chain(retriever)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")

user_input = st.text_input("Your message", key="chat_input", placeholder="Type your question here...")

if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""

if user_input and user_input != st.session_state.last_user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    response = qa_chain({"query": user_input})
    answer = response["result"]

    # Print documents and context
    with st.expander("Retrieved Documents and Context", expanded=False):
        for i, doc in enumerate(response.get("source_documents", [])):
            st.markdown(f"**Document {i+1}:**")
            st.code(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.session_state.last_user_input = user_input
    st.rerun()


