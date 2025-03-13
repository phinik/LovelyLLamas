"""
This file was used to sift out files that a) do not contain a weather report or b) where the weather data does not start
at midnight.
"""

import argparse
import os
import json
import shutil

from typing import Dict, List

def list_files(path: str) -> List[str]:
    return [file for file in os.listdir(path) if "_standardised.json" in file]

def load_file(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def starts_at_0_o_clock(content: Dict) -> bool:
    times = content["times"]
    report_short = content["report_short"]
    if times[0] == "00 - 01 Uhr" and report_short is not None:
        return True

    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    files = list_files(args.input_dir)

    for file in files:
        content = load_file(os.path.join(args.input_dir, file))

        if starts_at_0_o_clock(content=content):
            shutil.copyfile(os.path.join(args.input_dir, file), os.path.join(args.output_dir, file))
