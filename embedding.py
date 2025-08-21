import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

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
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
    pinecone_api_key = os.getenv("pinecone_api_key")
    pinecone_cloud = "aws"
    pinecone_region = os.getenv("pinecone_environment", "us-east-1")
    index_name = "product-index"

    pc = Pinecone(api_key=pinecone_api_key)
    print(f"Connecting to Pinecone index: {pc.list_indexes().names()}")
    # Check if index exists
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # Ensure this matches your embedding dimension
            metric='cosine',
            spec=ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region)
        )
    else:
        print(f"Pinecone index '{index_name}' already exists.")
        # Optionally, check the dimension and warn if it does not match

    # Get the actual index object
    pinecone_index = pc.Index(index_name)

    # Pass the Pinecone index object to Langchain PineconeVectorStore
    vectorstore = PineconeVectorStore(index=pinecone_index, embedding=embeddings)
    #vectorstore.add_documents(split_docs)
    return vectorstore

