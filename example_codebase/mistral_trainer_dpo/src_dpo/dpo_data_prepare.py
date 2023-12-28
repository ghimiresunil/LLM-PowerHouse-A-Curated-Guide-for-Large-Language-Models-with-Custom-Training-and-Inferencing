import json
import pandas as pd
from sklearn.model_selection import train_test_split

def data_loader(filepath):
    with open(filepath, encoding='utf-8', errors='ignore') as json_data:
        data = json.load(json_data)
    prompt_list = [item['input'] for item in data]
    chosen_list = [item['actual_data'] for item in data]
    rejected_list = [item['predicted_output'] for item in data]
    result = {
        "prompt": prompt_list,
        "chosen": chosen_list,
        "rejected": rejected_list
    }

    # Create a DataFrame
    df = pd.DataFrame(result)

    # Split the data into training (80%) and testing (20%) sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save training data to CSV
    train_df.to_csv('./dpo_data/train_data.csv', index=False)

    # Save testing data to CSV
    test_df.to_csv('./dpo_data/test_data.csv', index=False)

if __name__ == '__main__':
    data_loader('eval_data/dpo_final_data.json')