import pandas as pd

def load_data(filepath="dataset100.csv"):
    return pd.read_csv(filepath)
