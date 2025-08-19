import pandas as pd
import logging
from langchain.tools import Tool

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

def qa_tool_func(question, qa_chain):
    logging.info(f"qa_tool_func called with question: {question}")
    df = pd.read_csv("dataset100.csv")
    for pid in df["pid"].astype(str):
        if pid in question:
            context = df[df["pid"].astype(str) == pid]["text"].values
            if len(context) > 0:
                return context[0]
    question_lower = question.lower()
    for name in df["product_name"].astype(str):
        name_lower = name.lower()
        if name_lower in question_lower or question_lower in name_lower:
            context = df[df["product_name"].astype(str).str.lower() == name_lower]["text"].values
            if len(context) > 0:
                return context[0]
    result = qa_chain({"query": question})["result"]
    return result

def get_tools(qa_chain):
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
    qa_tool = Tool.from_function(
        func=lambda q: qa_tool_func(q, qa_chain),
        name="product_qa",
        description="Answer questions about products using embedded product information, product code, or product name."
    )
    return [price_tool, desc_tool, qa_tool]
