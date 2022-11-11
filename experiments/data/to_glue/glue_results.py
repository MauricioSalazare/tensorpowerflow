import pandas as pd
from pathlib import Path

current_directory = Path().absolute()
csv_files = current_directory.glob("*.csv")
files = [x for x in csv_files if x.is_file()]

df_list = []
for file in files:
    df = pd.read_csv(file, index_col=[0])
    df["unique_id"] = df.index + "_" + df["time_steps"].astype(str) + "_" +  df["grid_size"].astype(str)
    df_list.append(df)
dataset = pd.concat(df_list, axis=0)

dataset = dataset.drop_duplicates(subset=["unique_id"], keep="first")
dataset.drop(columns=["unique_id"], inplace=True)

dataset = dataset.sort_values(by=["grid_size", "time_steps"]).reset_index()

dataset.to_csv(current_directory.parent/ "merged_data.csv", index=False)