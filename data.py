import ast
import csv
import json
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

# df = pd.read_json("data/alpaca_evol_instruct_70k.json")

wiz_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
INST_REPLACE_ME

### Response: OUT_REPLACE_ME"""

# data = []

# for i in range(len(df)):
#     inst = df.iloc[i]["instruction"]
#     output = df.iloc[i]["output"]

#     out_str = wiz_template.replace("INST_REPLACE_ME", inst).replace(
#         "OUT_REPLACE_ME", output
#     )

#     data.append({"text": out_str})

# df["data"] = data
# df.to_csv("data/evol_data.csv", index=False)

# data_list = []

# with open("data/CodeExercise-Python-27k.json") as f:
#     data = f.readlines()

#     for dict_item in data:
#         dict_item = ast.literal_eval(dict_item)["chat_rounds"]
#         inp_str = dict_item[0]["content"]
#         out_str = dict_item[1]["content"]

#         data_str = wiz_template.replace("INST_REPLACE_ME", inp_str).replace(
#             "OUT_REPLACE_ME", out_str
#         )
#         data_list.append(data_str)

# df = pd.DataFrame({"data": data_list})
# print(df.head())

# with open("data/codefusion_data.csv", mode="w", newline="", encoding="utf-8") as file:
#     writer = csv.writer(file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(df.columns)
#     for index, row in df.iterrows():
#         try:
#             writer.writerow(row)
#         except:
#             pass

df = pd.read_csv("MathInstruct.csv")
print(len(df))

# model_name_or_path = "lora/mathv1"
# tokenizer = AutoTokenizer.from_pretrained(
#     model_name_or_path, use_fast=True
# )

data, tok_lens = [], []
c = 0

for i in tqdm(range(len(df))):
    inst_str = str(df.iloc[i]["question"])
    out_str = str(df.iloc[i]["answer"])

    data_item = wiz_template.replace("INST_REPLACE_ME", inst_str).replace(
        "OUT_REPLACE_ME", out_str
    )
    # tok = len(tokenizer(data_item, return_tensors="pt").input_ids[0])
    # tok_lens.append(tok)
    data.append(data_item)

data = pd.DataFrame({"data": data})
data.to_csv("data/math/MathInstruct.csv", index=False)
