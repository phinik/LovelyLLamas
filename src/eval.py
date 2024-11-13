import argparse
import tqdm
import torch
import torch.nn as nn

from typing import Dict
from src.dataloader import get_eval_dataloader_weather_dataset
from src.dummy_tokenizer import DummyTokenizer
from src.loss import CELoss


class Evaluator:
    def __init__(self, config: Dict, device: torch.device):
        self._config = config
        self._device = device

        # Dataloader
        self._eval_dataloader = get_eval_dataloader_weather_dataset(
            path=self._config["dataset"], 
            batch_size=self._config["batch_size"],
            cached=self._config["cached"]
        )

        # Tokenizer
        self._tokenizer = DummyTokenizer(self._config["dataset"])

        # Loss
        self._loss = CELoss(ignore_idx=self._tokenizer.padding_idx_target)

    @torch.no_grad()
    def evaluate(self, model) -> Dict:
        model.eval()

        total_loss = 0
        total_loss_values = 0

        for i, batch in enumerate(tqdm.tqdm(self._eval_dataloader)):
            context = batch["overview"]
            targets = batch["report_short"]

            # Tokenize
            for i in range(len(context)):
                context[i] = torch.tensor(self._tokenizer.stoi_context(context[i])).unsqueeze(0)
                targets[i] = torch.tensor(self._tokenizer.stoi_targets("<start> " + targets[i] + " <stop>"))

            # Pad target sequences to have equal length and transform the list of tensors into a single tensor.
            targets = nn.utils.rnn.pad_sequence(
                targets, 
                padding_value=self._tokenizer.padding_idx_target, 
                batch_first=True
            )

            # Context sequences always have equal length, hence, no padding is required and the list of tensors is just
            # concatenated.
            context = torch.concat(context)

            # Move tensors 
            targets = targets.to(device=self._device)
            context = context.to(device=self._device)
            
            for i in range(0, targets.shape[1] - self._config["block_size"]):
                inputs = targets[:, i:i+self._config["block_size"]]
                labels = targets[:, i+1:i+1+self._config["block_size"]]

                prediction = model(context, inputs)
                
                total_loss_values += torch.sum(torch.where(labels != self._tokenizer.padding_idx_target, 1, 0))
                labels = labels.reshape(labels.shape[0] * labels.shape[1])  # B * T
                total_loss += self._loss(prediction, labels)
                
        return {"loss": total_loss / total_loss_values}
        
if __name__ == "__main__":
    evaluator = Evaluator()
    pass