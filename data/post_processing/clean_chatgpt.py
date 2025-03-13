"""
This file was used to clean the weather reports from both chatGPT datasets. The performed operations were derived from
the first chatGPT dataset. Applying this script to the second chatGPT dataset would probably not have been necessary,
however, it was applied to this dataset too to be on the safe sid.
"""

import argparse
import os
import json
import tqdm
import re

from typing import Dict, List


def list_files(path: str) -> List[str]:
    return [file for file in os.listdir(path) if "_standardised.json" in file]

def load_file(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)
    
def clean_chatgpt(content: Dict) -> Dict:
    if "gpt_rewritten_apokalyptisch_v2" not in content.keys():
        return content
    
    target = content["gpt_rewritten_apokalyptisch_v2"]

    target = target.replace(content["city"], "<city>")
        
    target = target.replace("‚", ",")
    target = target.replace("**", "")
    target = target.replace("*", "")
    target = target.replace("„", "\"")
    target = target.replace("“", "\"")
    target = target.replace("”", "\"")
    target = target.replace("‘", "'")
    target = target.replace("’", "'")
    target = target.replace("°C", "°C")
    target = re.sub(r"°( *)[C]?", r"°C", target)

    target = re.sub(r"([0-9]+)( +)°C", r"\1°C", target)
        
    target = target.replace("–", "—")
    target = target.replace(" - ", "—")
    target = target.replace("…", "")

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    
    space_pattern = re.compile(r"[ ]{2,}")
    newline_pattern = re.compile(r"[ ]+[\n]{2,}")
       
    target = emoji_pattern.sub(r'', target)
            
    if target.startswith("Hier") or target.startswith("Witziger"):
        reports = target.split("\n\n")
        if len(reports) != 1:
            if reports[0].strip().endswith(":") or reports[0].strip().endswith(content["city"]):
                target = "\n\n".join(reports[1:])

    target = newline_pattern.sub("\n\n", target)
    target = space_pattern.sub(' ', target)

    target = re.sub(r"( +)'([A-Za-z0-9]+)", r'\1 \"\2', target)
    target = re.sub(r"([A-Za-z0-9]+)'( +)", r'\1\" \2', target)
    target = re.sub(r"([0-9]+)-([0-9])+", r"\1 bis \2", target)
    target = re.sub(r"( +)-([A-Za-z]+)", r"-\2", target)
    target = re.sub(r"([A-Za-z]+)-( +)", r"\1-", target)
    target = re.sub(r"([a-z]+)'-([A-Za-z]+)", r'\1\"-\2', target)

    target = target.replace("´", "'")
    target = target.replace("Aben­teuer", "Abenteuer")
    target = target.replace("Regen­schauer", "Regenschauer")
    target = target.replace("grin­sen", "grinsen")
    target = target.replace("deli­kat­en", "delikaten")
    target = target.replace("<<", "\"")
    target = target.replace(">>", "\"")
    target = target.replace("\\", "\"")
    target = target.replace("_", "\"")
        
    target = target.replace('—', ' — ')

    target = re.sub(r'\"([A-Za-z0-9 ]+)\"-([A-Za-z0-9]+)', r'\1 \2', target)
           
    target = target.replace("<city>", content["city"])
    content["gpt_rewritten_apokalyptisch_v2"] = target.strip().strip("\"").strip()
        
    return content

def save_file(path: str, content: Dict) -> None:
    with open(path, "wb") as f:
        data_json = json.dumps(content, ensure_ascii=False, indent=4).encode("utf-8")
        f.write(data_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)

    args = parser.parse_args()

    files = list_files(args.dir)

    for file in tqdm.tqdm(files):
        content = load_file(os.path.join(args.dir, file))

        content = clean_chatgpt(content)

        save_file(os.path.join(args.dir, file), content)

