import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re

from typing import List, Dict


def load_filenames(dataset: str) -> List[str]:
    input_files = ["dset_train.json", "dset_eval.json", "dset_test.json"]
    
    files = []
    for file in input_files:
        path = os.path.join(dataset, file)
        with open(path, "r") as f:
            files += list(json.load(f))

    return files


def load_reports(dataset: str, files: List[str], key: str) -> List[str]:
    reports = []

    for file in files:
        path = os.path.join(dataset, file)
        
        with open(os.path.join(path), "r") as f:
            file_content = json.load(f)
            
        reports.append(get_prepped_report(file_content, key))

    return reports


def get_prepped_report(data: Dict, key: str) -> str:
    report = data[key]
    report = report.replace(data["city"], "<city>")
    report = report.replace("°C", "")
    
    report = re.sub(r"[.,;:?!]", r"", report)  # exclude punctuation
    report = re.sub(r"-?[0-9]+", r"<number>", report)  # exclude numbers

    return report


def count_occurances(reports: List[str]) -> Dict:
    idf_counts = {}  # count the number of documents a term t appears in
    tf_counts = []  # count the number of times a term t appears in a report

    for report in reports:
        # count the number of times a term t appears in the report
        report_tf_counts = {}
        words = report.split()
        for word in words:
            report_tf_counts[word] = report_tf_counts.get(word, 0) + 1
        tf_counts.append(report_tf_counts)
        
        # update term in documents counts
        for word in report_tf_counts.keys():
            idf_counts[word] = idf_counts.get(word, 0) + 1

    # for word, count in idf_counts.copy().items():
    #     if count < 5:
    #         idf_counts.pop(word, None)
    #         for report in tf_counts:
    #             report.pop(word, None)

    # print(len(idf_counts))

    return {"idf_counts": idf_counts, "tf_counts": tf_counts}


def calculate_similarity(counts: Dict) -> np.array:
    n_reports = len(counts["tf_counts"])  # total number of reports
    idfs = {key: np.log(n_reports / v) for key, v in counts["idf_counts"].items()}

    # keys must be in a fixed order in order for the vectors to have the same order
    sorted_keys = sorted(list(idfs.keys()))

    vocab_size = len(sorted_keys)

    # calculate tf-idf vectors
    vecs = np.zeros(shape=(n_reports, vocab_size))
    for i, report_counts in enumerate(counts["tf_counts"]):
        tf_norm = np.sum(list(report_counts.values()))  # length of report

        # normalize tf-counts
        tfs = {key: (v / tf_norm) for key, v in report_counts.items()}

        # calculate tf-idf values for each term in the vocab
        vec = np.zeros(shape=vocab_size)
        for j, key in enumerate(sorted_keys):
            vec[j] = tfs.get(key, 0) * idfs[key]

        # normalize vector to have unit length for later computation of cosine similarity
        vec = vec / np.sqrt(np.sum(np.power(vec, 2)))

        vecs[i, :] = vec

    similarities = np.zeros(shape=(n_reports, n_reports))
    for i in range(vecs.shape[0]):
        # pair-wise cosine similarity of report i with each report
        similarities[i, :] = np.matmul(vecs, vecs[i, :].T).T  

    return similarities


def calculate_mean_top_x_similarity(x: int, similarities: np.array) -> np.array:
    # diag elements are always 1 as they represent the cosine similarity of a report with itself. hence, we set the 
    # diagonal elements to zero.
    similarities = similarities - np.diag(np.diag(similarities))
    
    # sort similarities row-wise in ascending order
    sorted_sims = np.sort(similarities, axis=1)
    
    # compute the mean of the x highest similarity values for each report (excluding the similarity of the report with
    # itself)
    return np.mean(sorted_sims[:, -(x+1):-1], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--key", type=str, required=True)

    args = parser.parse_args()

    filenames = load_filenames(args.dataset)
    reports = load_reports(args.dataset, filenames, args.key)
    
    print("Counting...")
    counts = count_occurances(reports)
   
    print("Computing similarity...")
    similarities = calculate_similarity(counts)
    
    x = 50
    mean_top_x_similarity = calculate_mean_top_x_similarity(x, similarities)
    print(f"Mean top {x} similarity: ", np.mean(mean_top_x_similarity))
    
    np.save(os.path.join(args.dataset, f"{args.key}_top_{x}_similarity"), mean_top_x_similarity)

    plt.hist(mean_top_x_similarity, bins=np.arange(0, 1.025, 0.025))
    plt.ylabel("Counts", fontsize=18)
    plt.xlabel("Mean Top 50 Cosine Similarity (TF-IDF)", fontsize=18)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.show()
