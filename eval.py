import torch
import warnings
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")

# acrastt/Marx-3B-V2  microsoft/phi-1_5 lora/codefusion - 2 WizardLM/WizardCoder-3B-V1.0
model_name_or_path = "WizardLM/WizardCoder-3B-V1.0"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, device_map="cuda:0", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
)


prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Write python code to download image from a given url.

### Response: """

out_str = "Write python code to download video from youtube with the url as input."
prompt = prompt_template.replace("INST_REPLACE_ME", out_str)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
output = model.generate(inputs=input_ids, max_length=100)
output = tokenizer.decode(output[0])

print(output)
