import pandas as pd
from datasets import load_dataset, Dataset

def load_dataset_from_file(dataset_path):
    df = pd.read_csv(dataset_path)
    data = Dataset.from_pandas(df)
    return data

if __name__ == "__main__":
    data = load_dataset_from_file("dataset/final_df.csv")
