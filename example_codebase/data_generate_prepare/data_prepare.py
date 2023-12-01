import json
import pandas as pd


def remove_duplicates_from_json(file_path):
    with open(file_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data).drop("topic", axis=1)
    df = df.drop_duplicates(subset="output", keep="first")
    return df


def add_text_col(x):
    intro = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    instruction = " ### Instruction: " + x["instruction"]
    input = " ### Input: " + x["input"]
    respones = " ### Response: " + x["output"]
    return intro + instruction + input + respones


if __name__ == "__main__":
    dataset_path = "dataset/interview_dataset.json"
    result_df = remove_duplicates_from_json(dataset_path)
    result_df["text"] = result_df.apply(add_text_col, axis=1)
    # print(result_df.shape)
    result_df.to_csv("./dataset/final_df.csv", index=False)
