import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
import pandas as pd
import pinecone

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def prepare_documents(df):
    df["text"] = df.apply(
        lambda row: f"Product Code: {row['pid']}\nProduct: {row['product_name']}\nDescription: {row['description']}\nCategory: {row['product_category_tree']}\nPrice: ${row['retail_price']}",
        axis=1
    )
    docs = DataFrameLoader(df[["text"]], page_content_column="text").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

def get_faiss_vectorstore(df):
    split_docs = prepare_documents(df)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.from_documents(split_docs, embeddings)

def get_pinecone_vectorstore(df):
    split_docs = prepare_documents(df)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    pinecone_api_key = os.getenv("pinecone_api_key")
    pinecone_env = os.getenv("pinecone_environment")
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    index_name = "sample"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)
    return Pinecone.from_documents(split_docs, embeddings, index_name=index_name)

