import json
import numpy as np
import os

from abc import ABC, abstractmethod
from typing import Dict

from evaluate import load


class IMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get(self) -> float:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def update(self, prediction, label) -> None:
        pass


class BertScore(IMetric):
    def __init__(self, dataset: str):
        self._scorer = load("bertscore")
        
        self._scores_f1 = []  # F1 score
        self._scores_pre = []  # Precision
        self._scores_rec = []  # Recall

        self._idfs = self._load_idfs(dataset)

        # For some very weird reasons, the first call to compute(..) always results in a recall value that is 'nan' IFF
        # idf weights are provided. This seems to be independent of the actual predictions / references passed to 
        # compute(..). Hence, we call it once on dummy data to circumvent this issue.
        _ = self._scorer.compute(
            predictions=["In <city> scheint die Sonne"], 
            references=["In <city> scheint die Sonne"], 
            lang="de", 
            idf=self._idfs
        )

    @staticmethod
    def _load_idfs(dataset: str) -> Dict:
        with open(os.path.join(dataset, "idfs_of_words.json")) as f:
            return json.load(f)["eval"]

    @property
    def name(self) -> str:
        return "BertScore"
    
    def get(self) -> float:
        return {
            "mean_f1": np.mean(self._scores_f1),
            "mean_precision": np.mean(self._scores_pre),
            "mean_recall": np.mean(self._scores_rec)
        }

    def reset(self) -> None:
        self._scores_f1 = [] 
        self._scores_pre = []
        self._scores_rec = []

    def update(self, prediction: str, label: str) -> None:
        # TODO: move into postprocessing       
        prediction = prediction.replace(".", "")
        prediction = prediction.replace(",", "")
        prediction = prediction.replace("!", "")
        prediction = prediction.replace("<stop>", "")
        prediction = prediction.replace("  ", " ")
        prediction = prediction.strip()

        label = label.replace(".", "")
        label = label.replace(",", "")
        label = label.replace("!", "")
        label = label.replace("  ", " ")
        label = label.strip()

        tfs = {}
        for term in label.split():
            tfs[term] = tfs.get(term, 0) + 1

        tf_max = max(tfs.values())

        for term, tf in tfs.items():
            tfs[term] = tf / tf_max

        tf_idf = {}
        for term in tfs.keys():
            tf_idf[term] = tfs[term] * self._idfs[term]
    
        scores = self._scorer.compute(predictions=[prediction], references=[label], lang="de", idf=tf_idf)

        self._scores_f1.append(scores["f1"])
        self._scores_pre.append(scores["precision"])
        self._scores_rec.append(scores["recall"])   
