import argparse
import json
import numpy as np
import os
import matplotlib.pyplot as plt

from typing import List, Set


def load_reports(dataset: str) -> List[str]:
    reports = []

    dirs = ["train", "eval", "test"]
    for dir in dirs:
        dir_path = os.path.join(dataset, dir)
        
        files = os.listdir(dir_path)
        for file in files:
            with open(os.path.join(dir_path, file), "r") as f:
                file_content = json.load(f)
            
            report = file_content["report_short"]
            report = report.replace(file_content["city"], "<city>")
            report = report.replace("°C", " <degC>")
            report = report.replace("km/h", " <kmh>")
            report = report.replace(".", "")

            reports.append(report)

    return reports   


def determine_vocab(reports: List[str]) -> Set:
    vocab = set()

    for report in reports:
        contained_words = []

        words = report.split()
        for word in words:
            try:
                _ = float(word)  # exclude any numbers
            except ValueError:
                contained_words.append(word)
        contained_words = set(contained_words)

        vocab = vocab.union(contained_words)

    return vocab


def determine_report_lengths(reports: List[str]) -> List:
    lengths = []

    for report in reports:
        lengths.append(len(report.split()))

    return lengths


def load_cities(dataset: str) -> Set[str]:
    cities = []

    dirs = ["train", "eval", "test"]
    for dir in dirs:
        dir_path = os.path.join(dataset, dir)
        
        files = os.listdir(dir_path)
        for file in files:
            with open(os.path.join(dir_path, file), "r") as f:
                file_content = json.load(f)
            
            city = file_content["city"]
            
            cities.append(city)

    return set(cities)


def determine_report_sentence_lengths(dataset: str) -> List[int]:
    lengths = []

    dirs = ["train", "eval", "test"]
    for dir in dirs:
        dir_path = os.path.join(dataset, dir)
        
        files = os.listdir(dir_path)
        for file in files:
            with open(os.path.join(dir_path, file), "r") as f:
                file_content = json.load(f)
            
            report = file_content["report_short"]
            lengths.append(report.count("."))

    return lengths   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()

    reports = load_reports(args.dataset)
    print("Total number: ", len(reports))

    reports_lengths = determine_report_lengths(reports)
    print("Avg length: ", np.mean(reports_lengths))
    
    vocab = determine_vocab(reports)   
    print("Vocab size reports: ", len(vocab))

    cities = load_cities(args.dataset)
    print("# cities: ", len(cities))

    report_sentence_length = determine_report_sentence_lengths(args.dataset)
    print("Report avg sentence length: ", np.mean(report_sentence_length))

    plt.hist(reports_lengths, bins=20)
    plt.show()
