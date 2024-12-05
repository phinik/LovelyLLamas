import argparse
import json
import numpy as np
import os
import matplotlib.pyplot as plt

from typing import List, Dict


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
            report = report.replace("°C", "")
            report = report.replace("km/h", "")
            report = report.replace(".", "")

            reports.append(report)

    return reports    


def count_occurances(reports: List[str]) -> Dict:
    idf_counts = {}  # count the number of documents a term t appears in
    tf_counts = []  # count the number of times a term t appears in a report

    for report in reports:
        # count the number of times a term t appears in the report
        report_tf_counts = {}
        words = report.split()
        for word in words:
            try:
                _ = float(word)  # exclude any numbers
            except ValueError:
                report_tf_counts[word] = report_tf_counts.get(word, 0) + 1
        tf_counts.append(report_tf_counts)
        
        # update term in documents counts
        for word in report_tf_counts.keys():
            idf_counts[word] = idf_counts.get(word, 0) + 1

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


def calculate_mean_top_10_similarity(similarities: np.array) -> np.array:
    # diag elements are always 1 as they represent the cosine similarity of a report with itself. hence, we set the 
    # diagonal elements to zero.
    similarities = similarities - np.diag(np.diag(similarities))
    
    # sort similarities row-wise in ascending order
    sorted_sims = np.sort(similarities, axis=1)
    
    # compute the mean of the 10 highest similarity values for each report (excluding the similarity of the report with
    # itself)
    return np.mean(sorted_sims[:, -11:-1], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()

    reports = load_reports(args.dataset)
    
    print("counting")
    counts = count_occurances(reports)
   
    print("similarity")
    similarities = calculate_similarity(counts)
    
    mean_top_10_similarity = calculate_mean_top_10_similarity(similarities)
    print(mean_top_10_similarity)

    plt.hist(mean_top_10_similarity, bins=np.arange(0, 1.025, 0.025))
    print(np.mean(mean_top_10_similarity))
    #plt.imshow(np.log(similarities+1))
    #ax = plt.gca()
    #ax.set_yscale("log")
    plt.show()
