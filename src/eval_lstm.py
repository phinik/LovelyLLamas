import argparse
import tqdm
import torch
import torch.nn as nn
import os

from typing import Dict
from src.models.lstm import LSTM
from src.dataloader import get_eval_dataloader_weather_dataset
from src.tokenizer import TokenizerFactory, ContextTokenizer
from src.loss import CELoss
from src import utils
import json


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self, config: Dict):
        self._config = config

        # Dataloader
        self._eval_dataloader = get_eval_dataloader_weather_dataset(
            path=self._config["dataset"], 
            batch_size=self._config["batch_size"],
            num_workers=self._config["num_workers"],
            cached=self._config["cached"]
        )

        # Tokenizer
        self._context_tokenizer = ContextTokenizer(self._config["dataset"])
        self._target_tokenizer = TokenizerFactory.get(self._config["dataset"], self._config["tokenizer"], self._config["target"])

        # Loss
        self._loss = CELoss(ignore_idx=self._target_tokenizer.padding_idx)

        self._target_str = utils.TargetSelector.select(self._config["target"])
        self._overview_str = utils.OverviewSelector.select(self._config["overview"])

        print(f" [TARGET] {self._target_str.upper()}")
        print(f" [OVERVIEW] {self._overview_str.upper()}")

    @torch.no_grad()
    def evaluate(self, model) -> Dict:
        model.eval()

        total_loss = 0
        total_loss_values = 0

        for i, batch in enumerate(tqdm.tqdm(self._eval_dataloader)):
            context = batch[self._overview_str]
            targets = batch[self._target_str]

            # Tokenize
            for j in range(len(context)):
                context[j] = torch.tensor(self._context_tokenizer.stoi(context[j])).unsqueeze(0)
                targets[j] = torch.tensor(self._target_tokenizer.stoi(self._target_tokenizer.add_start_stop_tokens(targets[j])))

            # Pad target sequences to have equal length and transform the list of tensors into a single tensor.
            targets = nn.utils.rnn.pad_sequence(
                targets, 
                padding_value=self._target_tokenizer.padding_idx, 
                batch_first=True
            )

            # Context sequences always have equal length, hence, no padding is required and the list of tensors is just
            # concatenated.
            context = torch.concat(context)

            # Create batch
            batch = utils.batchify(context, targets, self._config["block_size"], DEVICE)

            for contexts, inputs, labels in zip(batch["context"], batch["inputs"], batch["labels"]):
                # Move tensors 
                contexts = contexts.to(device=DEVICE)
                inputs = inputs.to(device=DEVICE)
                labels = labels.to(device=DEVICE)
    
                prediction = model(contexts, inputs)
                
                total_loss_values += torch.sum(torch.where(labels != self._target_tokenizer.padding_idx, 1, 0))
                labels = labels.view(labels.shape[0] * labels.shape[1])  # B * T
                #labels[labels == self._target_tokenizer.unknown_idx] = self._target_tokenizer.padding_idx

                total_loss += self._loss(prediction, labels)
                
        return {"loss": (total_loss / total_loss_values).item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Path to dataset root")
    parser.add_argument("--model_weights", type=str, help="Which model weights to use")
    parser.add_argument("--cache_data", action="store_true", help="All data will be loaded into the RAM before training")
    parser.add_argument("--tokenizer", type=str, choices=["sow", "bert"], default="dummy", help="Which tokenizer to use for the report")
    parser.add_argument("--num_workers", type=int, default=4, help="How many workers to use for dataloading")
    parser.add_argument("--target", type=str, choices=["default", "gpt"], required=True, help="What to train on")
    parser.add_argument("--overview", type=str, choices=["full", "ctpc", "ctc", "ct", "tpwc"], default="full", required=True, help="What overview to use")

    
    args = parser.parse_args()
        
    config = {
        "dataset": args.dataset_path,
        "model_weights": args.model_weights,
        "model_params": os.path.join(os.path.dirname(args.model_weights), "params.json"),
        "cached": args.cache_data,
        "batch_size": 10,
        "block_size": 20,
        "tokenizer": args.tokenizer,
        "num_workers": args.num_workers,
        "target": args.target,
        "overview": args.overview
    }

    with open(config["model_params"], "r") as f:
        params = json.load(f)

    c = {k: v for k, v in params.items() if k in LSTM.__init__.__code__.co_varnames}
    model = LSTM(**c)
    model.load_weights_from(config["model_weights"])
    model.to(DEVICE)

    evaluator = Evaluator(config)
    res = evaluator.evaluate(model)

    print(res)