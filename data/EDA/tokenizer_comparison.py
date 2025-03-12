"""
This file was used to compare the tokensets of the SoW and the Bert tokenizer.
"""

import json
import re

from transformers import BertTokenizer


with open("dataset_2024_12_12_gpt/context_tokens.json", "r") as f:
    context = json.load(f)

with open("dataset_2024_12_12_gpt/target_tokens.json", "r") as f:
    sow = json.load(f)

with open("dataset_2024_12_12_gpt/rep_short_tokens_bert.json", "r") as f:
    bert = json.load(f)


tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
tokenizer.add_tokens(["<degC>", "<city>"], special_tokens=False)

bert_decoded = []

for token in bert:
    bert_decoded.append(tokenizer.decode(token, skip_special_tokens=True))

numbers = []
for token in bert_decoded:
    if re.match(r"-?[0-9]+", token):
        numbers.append(token)

print(len(set(sow)), len(sow))
print(len(set(bert_decoded)), len(bert_decoded))

set_sow = set(sow)
set_bert = set(bert_decoded)

intersection = set_sow.intersection(set_bert)
union = set_sow.union(set_bert)

print("IoU: ", len(intersection) / len(union))


non_numeric = []
for token in context:
    if not re.match(r"-?[0-9]+(.[0-9]+)?", token):
        non_numeric.append(token)

print(len(context), len(non_numeric))