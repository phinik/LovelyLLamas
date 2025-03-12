"""
This file was used to fill in the correct city names based on city_data.csv
"""

import argparse
import os
import json
import tqdm
import csv


from typing import Dict, List


def list_files(path: str) -> List[str]:
    return [file for file in os.listdir(path) if "_standardised.json" in file]


def load_city_data() -> Dict:
    with open("./data/misc/crawled_information/city_data.csv", "r") as f:
        reader = csv.DictReader(f)

        city_data = {}
        for item in reader:
            city_data[item["Identifier"]] = item

    return city_data


def load_file(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def fix_city_name(content: Dict, city_data: Dict) -> Dict:
    identifier = content["city"].split("-")[-1]
    
    content["city"] = city_data[identifier]["True Name"].strip()

    return content


def save_file(path: str, content: Dict) -> None:
    with open(path, "wb") as f:
        data_json = json.dumps(content, ensure_ascii=False, indent=4).encode("utf-8")
        f.write(data_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)

    args = parser.parse_args()

    files = list_files(args.dir)
    city_data = load_city_data()

    for file in tqdm.tqdm(files):
        content = load_file(os.path.join(args.dir, file))

        content_n = fix_city_name(content, city_data)

        save_file(os.path.join(args.dir, file), content)

