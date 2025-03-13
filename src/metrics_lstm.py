import numpy as np
import re
import torch

from abc import ABC, abstractmethod
from typing import List, Dict

from evaluate import load

from src.models.lstm import LSTM
from src.tokenizer import ContextTokenizer
from src.data_postprocessing import RemovePunctutation


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    def update(
        self, 
        prediction: str, 
        tokenized_prediction: List[int], 
        label: str, 
        contexts: Dict, 
        temperature: List[str]
    ) -> None:
        pass


class Rouge(IMetric):
    def __init__(self):
        self._scorer = load("rouge")
        
        self._scores_rouge_1 = []  # Rouge-1
        self._scores_rouge_2 = []  # Rouge-2
        self._scores_rouge_l = []  # Rouge-L

        self._transform = RemovePunctutation()

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

    def update(
        self, 
        prediction: str, 
        tokenized_prediction: List[int], 
        label: str, 
        contexts: Dict,
        temperature: List[str]
    ) -> None:
        prediction = self._transform(prediction)
        label = self._transform(label)
    
        scores = self._scorer.compute(predictions=[prediction], references=[label])

        self._scores_rouge_1.append(scores["rouge1"])
        self._scores_rouge_2.append(scores["rouge2"])
        self._scores_rouge_l.append(scores["rougeL"])


class Bleu(IMetric):
    def __init__(self):
        self._scorer = load("bleu")
        
        self._scores = []  # BLEU score

        self._transform = RemovePunctutation()

    @property
    def name(self) -> str:
        return "BLEU"
    
    def get(self) -> float:
        return {
            "mean_bleu": np.mean(self._scores),
        }

    def reset(self) -> None:
        self._scores = []

    def update(
        self, 
        prediction: str, 
        tokenized_prediction: List[int], 
        label: str, 
        contexts: Dict,
        temperature: List[str]
    ) -> None:
        prediction = self._transform(prediction)
        label = self._transform(label)
    
        scores = self._scorer.compute(predictions=[prediction], references=[label])

        self._scores.append(scores["bleu"])  


class BertScore(IMetric):
    def __init__(self):
        self._scorer = load("bertscore")
        
        self._scores_f1 = []  # F1 score
        self._scores_pre = []  # Precision
        self._scores_rec = []  # Recall

        self._transform = RemovePunctutation()

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

    def update(
        self, 
        prediction: str, 
        tokenized_prediction: List[int], 
        label: str, 
        contexts: Dict,
        temperature: List[str]
    ) -> None:
        prediction = self._transform(prediction)
        label = self._transform(label)

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

        self._transformation = RemovePunctutation()

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

    def update(
        self, 
        prediction: str, 
        tokenized_prediction: List[int], 
        label: str, 
        contexts: Dict,
        temperature: List[str]
    ) -> None:
        prediction = self._transformation(prediction)

        self._n_samples += 1
        
        if "<city>" in prediction:
            self._n_samples_with_city += 1

        
class TemperatureCorrectness(IMetric):
    def __init__(self):       
        self._n_correct_temp = 0
        self._n_incorrect_temp = 0

        self._transform = RemovePunctutation()

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

    def update(
        self, 
        prediction: str, 
        tokenized_prediction: List[int], 
        label: str, 
        contexts: Dict,
        temperature: List[str]
    ) -> None:
        prediction = self._transform(prediction)
        
        # Assumption: any numeric value in the report is a temperature value
        temps = re.findall(r"-?[0-9]+", prediction)

        for temp in temps:
            if temp in temperature:
                self._n_correct_temp += 1
            else:
                self._n_incorrect_temp += 1


class TemperatureRange(IMetric):
    def __init__(self):       
        self._scores = []

        self._transform = RemovePunctutation()

    @property
    def name(self) -> str:
        return "TemperatureRange"
    
    def get(self) -> float:
        return {
            "mean_score": np.mean(self._scores)
        }

    def reset(self) -> None:
        self._scores = []

    def update(
        self, 
        prediction: str, 
        tokenized_prediction: List[int], 
        label: str, 
        contexts: Dict,
        temperature: List[str]
    ) -> None:
        prediction = self._transform(prediction)

        # "relative intersection" inspired by IoU
        # Weather reports are not required to state the overall daily minimum and maximum temperature. Hence, IoU 
        # does not make much sense here because the interval spanned by the context temperatures might be bigger than 
        # the interval spanned by the temperatures stated in the reports. This is NOT a bug.
        # 
        # We therefore compute "intersection over predicted interval" in order to quantify to what extend the predicted
        # interval is contained in the interval provided by the context. The score will be:
        # - 1 if the predicted interval is fully contained in the context interval
        # - 0 if both intervals do not intersect
        # - between (0, 1) otherwise

        # Assumption: any numeric value in the report is a temperature value
        # Find all numeric values in the report and convert them to integer
        pred_temps = re.findall(r"-?[0-9]+", prediction)
        pred_temps = [int(t) for t in pred_temps]

        # Convert all context temperature values to integer
        context_temps = [int(t) for t in temperature if t != "<missing>"]
        
        if len(pred_temps) == 0 or len(context_temps) == 0:  # No temperature values is considered as not intersecting
            self._scores.append(0)
        else:
            # Get min and max from the prediction
            pred_temp_min = np.min(pred_temps)
            pred_temp_max = np.max(pred_temps)
            
            # Get min and max from the context
            context_temp_min = np.min(context_temps)
            context_temp_max = np.max(context_temps)          

            numerator = max(0, min(context_temp_max, pred_temp_max) - max(context_temp_min, pred_temp_min) + 1)
            denominator = pred_temp_max - pred_temp_min + 1

            self._scores.append(numerator / denominator)


class CustomClassifier(IMetric):
    def __init__(self, dataset_path=str): 
        self._scores = []

        model_params = "./checkpoints/final_classifier_dmodel_64_2024_12_12_bert/params.json"
        model_weights = "./checkpoints/final_classifier_dmodel_64_2024_12_12_bert/best_model_CE_loss.pth"
        self._block_size = 20

        c = {k: v for k, v in c.items() if k in LSTM.__init__.__code__.co_varnames}
        self._model = LSTM(**c)
        self._model.load_weights_from(model_weights)
        self._model.to(DEVICE)

        # Tokenizer
        self._context_tokenizer = ContextTokenizer(dataset_path)

    @property
    def name(self) -> str:
        return "CustomClassifierFull"
    
    def get(self) -> float:
        return {
            "mean_score": np.mean(self._scores)
        }

    def reset(self) -> None:
        self._scores = []

    def update(
        self, 
        prediction: str, 
        tokenized_prediction: List[int], 
        label: str, 
        contexts: Dict,
        temperature: List[str]
    ) -> None:
        tokenized_context = self._context_tokenizer.stoi(contexts["overview_full"])

        # Get sequences of size 'block_size'
        pred_seq = []
        context_seq = []

        if len(tokenized_prediction) < self._block_size:
            pred_seq.append(torch.tensor(tokenized_prediction).unsqueeze(0))
            context_seq.append(torch.tensor(tokenized_context).unsqueeze(0))
        else:
            for i in range(0, len(tokenized_prediction) - self._block_size + 1):
                pred_seq.append(torch.tensor(tokenized_prediction[i:i+self._block_size]).unsqueeze(0))
                context_seq.append(torch.tensor(tokenized_context).unsqueeze(0))
      
        # Create a single batch of sequences
        inputs = torch.cat(pred_seq)
        contexts = torch.cat(context_seq)

        # Move batch
        inputs = inputs.to(DEVICE)
        contexts = contexts.to(DEVICE)

        # Classify batch
        outputs = self._model(contexts, inputs)
        outputs = torch.nn.functional.sigmoid(outputs)

        # Compute and store score
        self._scores.append(torch.mean(outputs).item())     


class CustomClassifierCT(IMetric):
    def __init__(self, dataset_path=str): 
        self._scores = []

        model_params = "./checkpoints/final_classifier_dmodel_64_2024_12_12_bert_ct/params.json"
        model_weights = "./checkpoints/final_classifier_dmodel_64_2024_12_12_bert_ct/best_model_CE_loss.pth"
        self._block_size = 20

        c = {k: v for k, v in c.items() if k in LSTM.__init__.__code__.co_varnames}
        self._model = LSTM(**c)
        self._model.load_weights_from(model_weights)
        self._model.to(DEVICE)

        # Tokenizer
        self._context_tokenizer = ContextTokenizer(dataset_path)

    @property
    def name(self) -> str:
        return "CustomClassifierCT"
    
    def get(self) -> float:
        return {
            "mean_score": np.mean(self._scores)
        }

    def reset(self) -> None:
        self._scores = []

    def update(
        self, 
        prediction: str, 
        tokenized_prediction: List[int], 
        label: str, 
        contexts: Dict,
        temperature: List[str]
    ) -> None:
        tokenized_context = self._context_tokenizer.stoi(contexts["overview_ct"])

        # Get sequences of size 'block_size'
        pred_seq = []
        context_seq = []

        if len(tokenized_prediction) < self._block_size:
            pred_seq.append(torch.tensor(tokenized_prediction).unsqueeze(0))
            context_seq.append(torch.tensor(tokenized_context).unsqueeze(0))
        else:
            for i in range(0, len(tokenized_prediction) - self._block_size + 1):
                pred_seq.append(torch.tensor(tokenized_prediction[i:i+self._block_size]).unsqueeze(0))
                context_seq.append(torch.tensor(tokenized_context).unsqueeze(0))
      
        # Create a single batch of sequences
        inputs = torch.cat(pred_seq)
        contexts = torch.cat(context_seq)

        # Move batch
        inputs = inputs.to(DEVICE)
        contexts = contexts.to(DEVICE)

        # Classify batch
        outputs = self._model(contexts, inputs)
        outputs = torch.nn.functional.sigmoid(outputs)

        # Compute and store score
        self._scores.append(torch.mean(outputs).item())
        