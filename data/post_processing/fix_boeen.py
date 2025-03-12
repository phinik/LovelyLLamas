"""
The original weather reports from wetter.com sometimes contained a final sentence regarding gusts. However, there was no
gust data available on the website, so that an potential final gust sentence is removed from the reports using this
script.
"""

import argparse
import os
import json
import tqdm

from typing import Dict, List


def list_files(path: str) -> List[str]:
    return [file for file in os.listdir(path) if "_standardised.json" in file]

def load_file(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)
    
def fix_boeen(content: Dict) -> Dict:
    report_short = content["report_short"]
    
    sentences = [s for s in report_short.split(".") if "Böe" not in s]

    report_new = ".".join(sentences)

    content["report_short_wout_boeen"] = report_new
        
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

        content = fix_boeen(content)

        save_file(os.path.join(args.dir, file), content)

