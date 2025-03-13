"""
This script was used to fix issues with the values of the wind direction attributes. Some values contained an additional
random number which was removed using this script.
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
    
def fix_windrichtung(content: Dict) -> Dict:
    windrichtung = content["windrichtung"]

    for i in range(len(windrichtung)):
        if windrichtung[i] is not float("nan"):
            try:
                windrichtung[i] = windrichtung[i].split()[0]
            except:
                print(windrichtung[i])

        content["windrichtung"] = windrichtung

    return content

def fix_windspeed_unit(content: Dict) -> Dict:
    if "windgeschwindigkeit_in_km_per_s" in content.keys():
        content["windgeschwindigkeit_in_km_per_h"] = content["windgeschwindigkeit_in_km_per_s"]
        _ = content.pop("windgeschwindigkeit_in_km_per_s", None)
    
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

        content = fix_windrichtung(content)
        content = fix_windspeed_unit(content)

        save_file(os.path.join(args.dir, file), content)

