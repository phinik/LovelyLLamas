import numpy as np
import re

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

    def update(self, prediction: str, label: str, context: Dict) -> None:
        # TODO: move into postprocessing       
        prediction = prediction.replace(".", "")
        prediction = prediction.replace(",", "")
        prediction = prediction.replace("!", "")
        prediction = prediction.replace("<stop>", "")
        prediction = re.sub(f"[ ]{2,}", r" ", prediction)  # make sure that there are only single spaces
        prediction = prediction.strip()

        label = label.replace(".", "")
        label = label.replace(",", "")
        label = label.replace("!", "")
        label = re.sub(f"[ ]{2,}", r" ", label)  # make sure that there are only single spaces
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

    def update(self, prediction: str, label: str, context: Dict) -> None:
        # TODO: move into postprocessing       
        prediction = prediction.replace(".", "")
        prediction = prediction.replace(",", "")
        prediction = prediction.replace("!", "")
        prediction = prediction.replace("<stop>", "")
        prediction = re.sub(f"[ ]{2,}", r" ", prediction)  # make sure that there are only single spaces
        prediction = prediction.strip()

        label = label.replace(".", "")
        label = label.replace(",", "")
        label = label.replace("!", "")
        label = re.sub(f"[ ]{2,}", r" ", label)  # make sure that there are only single spaces
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

    def update(self, prediction: str, label: str, context: Dict) -> None:
        # TODO: move into postprocessing       
        prediction = prediction.replace(".", "")
        prediction = prediction.replace(",", "")
        prediction = prediction.replace("!", "")
        prediction = prediction.replace("<stop>", "")
        prediction = re.sub(f"[ ]{2,}", r" ", prediction)  # make sure that there are only single spaces
        prediction = prediction.strip()

        label = label.replace(".", "")
        label = label.replace(",", "")
        label = label.replace("!", "")
        label = re.sub(f"[ ]{2,}", r" ", label)  # make sure that there are only single spaces
        label = label.strip()

        # Although stated differently in the documentation, a pre-computed idf-dict cannot be used. It's values are
        # simply ignored and it appears that the scorer computes idf-scores itself and uses these values instead.
        # Hence, no matter what idf-dict is provided, the same scores result and these scores are identical to the ones
        # obtained by simply setting idf=True. Since only one reference sentence is provided each time, we do not use 
        # idf-scores because they are only computed based in this one single reference.
        #
        # 'lang="de"' is equal to 'model_type="bert-base-multilingual-cased"'
        # (at least the resulting scores are the same)
        scores = self._scorer.compute(predictions=[prediction], references=[label], lang="de")

        self._scores_f1.append(scores["f1"])
        self._scores_pre.append(scores["precision"])
        self._scores_rec.append(scores["recall"])   


class CityAppearance(IMetric):
    def __init__(self):       
        self._n_samples = 0
        self._n_samples_with_city = 0

    @property
    def name(self) -> str:
        return "CityAppearance"
    
    def get(self) -> float:
        return {
            "accuracy": self._n_samples_with_city / self._n_samples
        }

    def reset(self) -> None:
        self._n_samples = 0
        self._n_samples_with_city = 0

    def update(self, prediction: str, label: str, context: Dict) -> None:
        # TODO: move into postprocessing       
        prediction = prediction.replace(".", "")
        prediction = prediction.replace(",", "")
        prediction = prediction.replace("!", "")
        prediction = prediction.replace("<stop>", "")
        prediction = re.sub(f"[ ]{2,}", r" ", prediction)  # make sure that there are only single spaces
        prediction = prediction.strip()

        label = label.replace(".", "")
        label = label.replace(",", "")
        label = label.replace("!", "")
        label = re.sub(f"[ ]{2,}", r" ", label)  # make sure that there are only single spaces
        label = label.strip()

        self._n_samples += 1
        
        if "<city>" in prediction:
            self._n_samples_with_city += 1

        
class TemperatureCorrectness(IMetric):
    def __init__(self):       
        self._n_correct_temp = 0
        self._n_incorrect_temp = 0

    @property
    def name(self) -> str:
        return "TemperatureCorrectness"
    
    def get(self) -> float:
        return {
            "accuracy": self._n_correct_temp / (self._n_correct_temp + self._n_incorrect_temp)
        }

    def reset(self) -> None:
        self._n_correct_temp = 0
        self._n_incorrect_temp = 0

    def update(self, prediction: str, label: str, context: Dict) -> None:
        # TODO: move into postprocessing       
        prediction = prediction.replace(".", "")
        prediction = prediction.replace(",", "")
        prediction = prediction.replace("!", "")
        prediction = prediction.replace("<stop>", "")
        prediction = re.sub(f"[ ]{2,}", r" ", prediction)  # make sure that there are only single spaces
        prediction = prediction.strip()

        temps = re.findall(r"[0-9]+", prediction)

        for temp in temps:
            if temp in context: #["temperatur_in_deg_C"]:
                self._n_correct_temp += 1
            else:
                self._n_incorrect_temp += 1
        