import argparse
import tqdm
import torch
import torch.nn as nn

from typing import Dict
from src.dataloader import get_eval_dataloader_weather_dataset
from src.dummy_tokenizer import DummyTokenizer
from src.loss import CELoss
from src.models import Transformer
from src.tokenizer import Tokenizer
import src.determinism


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self, config: Dict):
        self._config = config

        # Dataloader
        self._eval_dataloader = get_eval_dataloader_weather_dataset(
            path=self._config["dataset"], 
            batch_size=self._config["batch_size"],
            cached=self._config["cached"]
        )

        # Tokenizer
        self._context_tokenizer = DummyTokenizer(self._config["dataset"])
        self._target_tokenizer = DummyTokenizer(self._config["dataset"]) if self._config["tokenizer"] == "dummy" else Tokenizer()

        # Loss
        self._loss = CELoss(ignore_idx=self._target_tokenizer.padding_idx)

    @torch.no_grad()
    def evaluate(self, model) -> Dict:
        model.eval()

        total_loss = 0
        total_loss_values = 0

        for i, batch in enumerate(tqdm.tqdm(self._eval_dataloader)):
            context = batch["overview"]
            targets = batch["report_short"]

            # Tokenize
            for j in range(len(context)):
                context[j] = torch.tensor(self._context_tokenizer.stoi_context(context[j])).unsqueeze(0)
                targets[j] = torch.tensor(self._target_tokenizer.stoi("<start> " + targets[j] + " <stop>"))

            # Pad target sequences to have equal length and transform the list of tensors into a single tensor.
            targets = nn.utils.rnn.pad_sequence(
                targets, 
                padding_value=self._target_tokenizer.padding_idx, 
                batch_first=True
            )

            # Context sequences always have equal length, hence, no padding is required and the list of tensors is just
            # concatenated.
            context = torch.concat(context)

            # Move tensors 
            targets = targets.to(device=DEVICE)
            context = context.to(device=DEVICE)
            
            for j in range(0, targets.shape[1] - self._config["block_size"]):
                inputs = targets[:, j:j+self._config["block_size"]]
                labels = targets[:, j+1:j+1+self._config["block_size"]]

                prediction = model(context, inputs)
                
                total_loss_values += torch.sum(torch.where(labels != self._target_tokenizer.padding_idx, 1, 0))
                labels = labels.reshape(labels.shape[0] * labels.shape[1])  # B * T
                total_loss += self._loss(prediction, labels)
                
        return {"loss": (total_loss / total_loss_values).item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Path to dataset root")
    parser.add_argument("--model_weights", type=str, help="Which model weights to use")
    parser.add_argument("--model_params", type=str, help="Which model params to use")
    #parser.add_argument("--model", type=str, choices=["transformer", "lstm"], help="Which model to use")
    parser.add_argument("--cache_data", action="store_true", help="All data will be loaded into the RAM before training")
    parser.add_argument("--tokenizer", type=str, choices=["dummy", "bert"], default="dummy", help="Which tokenizer to use for the report")
    
    args = parser.parse_args()
        
    config = {
        "dataset": args.dataset_path,
        "model_weights": args.model_weights,
        "model_params": args.model_params,
        "cached": args.cache_data,
        "model": "transformer", #args.model,
        "batch_size": 5,
        "block_size": 20,
        "tokenizer": args.tokenizer
    }

    model = Transformer.from_params(config["model_params"])
    model.load_weights_from(config["model_weights"])
    model.to(DEVICE)

    evaluator = Evaluator(config)
    res = evaluator.evaluate(model)

    print(res)
