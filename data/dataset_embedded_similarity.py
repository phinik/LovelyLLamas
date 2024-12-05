import argparse
import json
import numpy as np
import os
import matplotlib.pyplot as plt

from typing import List
from sentence_transformers import SentenceTransformer


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

            reports.append(report)

    return reports    


def calculate_similarity(reports: List[str]) -> np.array:
    # 1. Load a pretrained Sentence Transformer model
    model = SentenceTransformer("all-MiniLM-L12-v2")
    #model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

    # 2. Calculate embeddings by calling model.encode()
    embeddings = model.encode(reports)

    # 3. Calculate the embedding similarities
    similarities = model.similarity(embeddings, embeddings)

    return similarities.cpu().numpy()


def calculate_mean_similarity(similarities: np.array) -> float:
    sims_tril = np.tril(similarities, k=-1)

    n = np.sum(np.tril(np.ones_like(similarities), k=-1))

    return np.sum(sims_tril) / n


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

    similarities = calculate_similarity(reports)

    mean_similarity = calculate_mean_similarity(similarities)
    print(mean_similarity)

    mean_top_10_similarity = calculate_mean_top_10_similarity(similarities)
    print(np.min(mean_top_10_similarity))
    print(np.mean(mean_top_10_similarity))
    plt.hist(mean_top_10_similarity, bins=np.arange(0, 1.025, 0.025))
    plt.show()
