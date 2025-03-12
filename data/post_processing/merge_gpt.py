""" 
This file was used to merge the with chatGPT rewritten weather reports into an existing dataset.
"""

import json
import os

source_path = "/media/philipp/Scratch/LovelyLLamas/apokalyptisch/files_for_chatGPT/2024-12-12"

dest_path = "/media/philipp/Scratch/LovelyLLamas/dataset_2024_12_12_apo"

with open(os.path.join(dest_path, "dset_eval.json"), "r") as f:
    files = json.load(f)
    
counter = 0
for file in files:
    source = os.path.join(source_path, os.path.basename(file))
    dest = os.path.join(dest_path, file)
    
    with open(source, "r") as f:
        src_data = json.load(f)
    
    with open(dest, "r") as f:
        dst_data = json.load(f)
    
    try:
        dst_data["gpt_rewritten_apo"] = src_data["gpt_rewritten_apokalyptisch_v2"]
        dst_data["gpt_rewritten_apo"] = dst_data["gpt_rewritten_apo"].replace("<city>", dst_data["city"])
    except KeyError:
        print(file)
        counter += 1
    
    with open(dest, "wb") as f:
        encoded = json.dumps(dst_data, ensure_ascii=False, indent=4).encode("utf-8")
        f.write(encoded)
        
print(counter, len(files))
