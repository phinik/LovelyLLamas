import numpy as np

from abc import ABC, abstractmethod

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


class Rouge(IMetric):
    def __init__(self):
        self._scorer = load("rouge")
        
        self._scores_rouge_1 = []  # Rouge-1
        self._scores_rouge_2 = []  # Rouge-2
        self._scores_rouge_l = []  # Rouge-L

    @property
    def name(self) -> str:
        return "ROUGE"
    
    def get(self) -> float:
        return {
            "mean_rouge_1": np.mean(self._scores_rouge_1),
            "mean_rouge_2": np.mean(self._scores_rouge_2),
            "mean_rouge_L": np.mean(self._scores_rouge_l),
        }

    def reset(self) -> None:
        self._scores_rouge_1 = []
        self._scores_rouge_2 = []
        self._scores_rouge_l = []

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
    
        scores = self._scorer.compute(predictions=[prediction], references=[label])

        self._scores_rouge_1.append(scores["rouge1"])
        self._scores_rouge_2.append(scores["rouge2"])
        self._scores_rouge_l.append(scores["rougeL"])


class Bleu(IMetric):
    def __init__(self):
        self._scorer = load("bleu")
        
        self._scores = []  # BLEU score

    @property
    def name(self) -> str:
        return "BLEU"
    
    def get(self) -> float:
        return {
            "mean_bleu": np.mean(self._scores),
        }

    def reset(self) -> None:
        self._scores = []

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
    
        scores = self._scorer.compute(predictions=[prediction], references=[label])

        self._scores.append(scores["bleu"])  


class BertScore(IMetric):
    def __init__(self):
        self._scorer = load("bertscore")
        
        self._scores_f1 = []  # F1 score
        self._scores_pre = []  # Precision
        self._scores_rec = []  # Recall

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

        # Although stated differently in the documentation, a pre-computed idf-dict cannot be used. It's values are
        # simply ignored and it appears that the scorer computes idf-scores itself and uses these values instead.
        # Hence, no matter what idf-dict is provided, the same scores result and these scores are identical to the ones
        # obtained by simply setting idf=True. Since only one reference sentence is provided each time, we do not use 
        # idf-scores because they are only computed based in this one single reference.
        scores = self._scorer.compute(predictions=[prediction], references=[label], lang="de")

        self._scores_f1.append(scores["f1"])
        self._scores_pre.append(scores["precision"])
        self._scores_rec.append(scores["recall"])   
