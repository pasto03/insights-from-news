import torch
import json
import pickle
from transformers import AutoTokenizer, DataCollatorWithPadding


# load raw dataset
merged_json = json.load(open('news_and_prices.json', 'rb'))

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir="tokenizer/")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

days_before = 30
days_prediction = 14

title_max_length = 40
content_max_length = 120

train_batch_size = 24


print("Config data loaded.")
