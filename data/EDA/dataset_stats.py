import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re

from typing import List, Set, Dict


def load_data(dataset: str, key: str) -> List[Dict]:
    split_files = ["dset_train.json", "dset_eval.json", "dset_test.json"]
    
    files = []
    for split_file in split_files:
        with open(os.path.join(dataset, split_file), "r") as f:
            files_in_split = json.load(f)
            files += files_in_split
            
    data = []
    for file in files:
        with open(os.path.join(dataset, file), "r") as f:
            content = json.load(f)
            
            if key in content.keys():
                data.append(content)
            
    return data   


def get_prepped_report(data: Dict, key: str) -> str:
    report = data[key]
    report = report.replace(data["city"], "<city>")
    report = report.replace("°C", " <degC>")
    
    report = re.sub(r"[.,;:?!]", r"", report)

    return report


def vocab(data: List[Dict], key: str) -> Set:
    vocab = set()

    for point in data:
        report = get_prepped_report(point, key)
        
        word_candidates = report.split()
        
        contained_words = []
        for word in word_candidates:
            if not re.match(r"-?[0-9]+", word):  # do not include numbers
                contained_words.append(word)
        contained_words = set(contained_words)

        vocab = vocab.union(contained_words)

    return vocab


def report_lengths(data: List[Dict], key: str) -> List:
    lengths = []

    for point in data:
        report = get_prepped_report(point, key)

        lengths.append(len(report.split()))
        
    return lengths


def cities(data: List[Dict]) -> Set[str]:
    cities = []

    for point in data:  
        cities.append(point["city"])

    return set(cities)


def number_of_sentences(data: List[Dict], key: str) -> List[int]:
    n_sentences = []

    for point in data:
        report = point[key]

        n_sentences.append(len(re.findall(r"[A-Za-z0-9]+[.!?]{1,1}", report)))

    return n_sentences   


def nans_in_context(data: List[Dict]) -> float:
    keys = [
        "clearness", "temperatur_in_deg_C", "niederschlagsrisiko_in_perc", 
        "niederschlagsmenge_in_l_per_sqm", "windrichtung", "windgeschwindigkeit_in_km_per_h", "bewölkungsgrad"
    ]
    
    n_tot = 0
    n_nan = 0
    for point in data:                              
        for key in keys:
            n_tot += len(point[key])

            for v in point[key]:
                try:
                    if np.isnan(v):
                        n_nan += 1
                    else:
                        continue
                except TypeError:
                    continue
            
    return n_nan / n_tot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--key", type=str, required=True)

    args = parser.parse_args()

    data = load_data(args.dataset, args.key)
    print("Total number: ", len(data))

    reports_lengths = report_lengths(data, args.key)
    print("Avg length: ", np.mean(reports_lengths))
    
    vocab = vocab(data, args.key)   
    print("Vocab size: ", len(vocab))

    cities = cities(data)
    print("# Cities: ", len(cities))

    number_of_sentences = number_of_sentences(data, args.key)
    print("Avg # sentences: ", np.mean(number_of_sentences))

    nans_in_context = nans_in_context(data)
    print("% Nans in context: ", nans_in_context * 100)

    stats_dict = {
        "key": args.key,
        "number_of_files": len(data),
        "avg_length": np.mean(reports_lengths),
        "vocab_size": len(vocab),
        "n_cities": len(cities),
        "avg_n_sentences": np.mean(number_of_sentences),
        "perc_nans_in_context": nans_in_context * 100
    }

    with open(os.path.join(args.dataset, f"{args.key}_stats.json"), 'w') as f:
        json.dump(stats_dict, f, indent=4)

    np.savez(os.path.join(args.dataset, f"{args.key}_stats"), lengths=reports_lengths, n_sentences=number_of_sentences)

    plt.hist(reports_lengths, bins=20)
    plt.show()
