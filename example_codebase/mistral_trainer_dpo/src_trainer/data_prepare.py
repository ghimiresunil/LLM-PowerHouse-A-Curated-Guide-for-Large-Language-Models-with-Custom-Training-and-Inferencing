import numpy as np
import pandas as pd


def clean_csv(file_path):
    json_df = pd.read_json('dataset/med_doc_patient.json')[['input', 'output']]
    df = pd.read_csv(file_path)
    df = df[~df.duplicated()]
    df = df.dropna(subset=["focus_area", "answer"])
    df = df[df.apply(lambda row: row["focus_area"] in row["question"], axis=1)]
    df = df.drop(columns=["source", "focus_area"])
    df = df.rename(columns={"question": "input", "answer": "output"})
    df = pd.concat([json_df, df])
    df[
        "instruction"
    ] = "If you are a doctor, please answer the medical questions based on the patient's description"
    df = df.reset_index(drop=True)
    return df


def add_text_col(df):
    intro = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    instruction = " ### Instruction: " + df["instruction"]
    input = " ### Input: " + df["input"]
    respones = " ### Response: " + df["output"]
    return intro + instruction + input + respones


if __name__ == "__main__":
    result_df = clean_csv("dataset/medquad.csv")
    result_df["text"] = result_df.apply(add_text_col, axis=1)
    result_df = result_df[["instruction", "input", "output", "text"]]
    print("Shape of final pre-processed data:", result_df.shape)
    result_df.to_csv("dataset/final_df.csv", index=False)
