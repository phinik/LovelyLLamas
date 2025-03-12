"""
This file was used to sift out files where the weather reports state temperature values that are not contained in the
weather data.
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
    
def good_temp(content: Dict) -> bool:
    report_short = content["report_short_wout_boeen"]
    report_short = report_short.replace("°C", "")
    report_short = report_short.replace(".", "")
    report_short = report_short.split()
    numbers = []
    for w in report_short:
        try:
            _ = float(w)
            numbers.append(w)
        except ValueError:
            continue

    temps = content["temperatur_in_deg_C"]

    outlier = 0
    for i, n in enumerate(numbers):
        if n not in temps:
            outlier += 1

    return True if outlier == 0 else False

def save_file(path: str, content: Dict) -> None:
    with open(path, "wb") as f:
        data_json = json.dumps(content, ensure_ascii=False, indent=4).encode("utf-8")
        f.write(data_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)

    args = parser.parse_args()

    files = list_files(args.dir)

    filtered_files = []
    for file in tqdm.tqdm(files):
        content = load_file(os.path.join(args.dir, file))

        if good_temp(content):
            filtered_files.append(file)
    
    print("[N FILES] ", len(filtered_files))
    
    with open("filtered_files.json", "w") as f:
        json.dump(filtered_files, f)