import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

def read_csv_from_dataset(filename, dataset_folder='dataset'):
    """
    Reads a CSV file from the specified dataset folder.

    Args:
        filename (str): Name of the CSV file.
        dataset_folder (str): Folder where datasets are stored.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    filepath = os.path.join(dataset_folder, filename)
    return pd.read_csv(filepath)


def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return docs