import argparse
import json
import os
import tqdm
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from typing import Dict

from src.best_model import BestModel, OptDirection
from src.dataloader import *
from src.loss import CELoss
from src.models import Transformer
from src.dummy_tokenizer import DummyTokenizer
from src.eval import Evaluator

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Generator:
    def __init__(self, config: Dict):
        self._config = config

        # Dataloader
        self._test_dataloader = get_test_dataloader_weather_dataset(
            path=self._config["dataset"], 
            batch_size=1,
            cached=self._config["cached"]
        )

        # Tokenizer
        self._tokenizer = DummyTokenizer(self._config["dataset"])

        self._model = Transformer(
        n_src_vocab=self._tokenizer.size_context_vocab, 
        n_trg_vocab=self._tokenizer.size_target_vocab, 
        src_pad_idx=self._tokenizer.padding_idx_context, 
        trg_pad_idx=self._tokenizer.padding_idx_target,
        emb_src_trg_weight_sharing=False,
        n_head=4,
        n_layers=2,
        d_inner=512
        )

        self._model.load_from(self._config["model"])
        
        self._model.to(DEVICE)
            

        # Tokenizer
        self._tokenizer = DummyTokenizer(self._config["dataset"])

    def sample(self):
        self._model.eval()

        for i, batch in enumerate(tqdm.tqdm(self._train_dataloader)):
            context = batch["overview"]
            targets = batch["report_short"]

            # Tokenize
            for i in range(len(context)):
                context[i] = torch.tensor(self._tokenizer.stoi_context(context[i])).unsqueeze(0)
                targets[i] = torch.tensor(self._tokenizer.stoi_targets("<start> " + targets[i] + " <stop>"))

            context = context[0]
            targets = targets[0]

            # Move tensors 
            targets = targets.to(device=DEVICE)
            context = context.to(device=DEVICE)
            
            running_input = torch.zeros(size=(1, self._config["block_size"] + 1))
            running_input[0, 0] = self._tokenizer.start_idx_target
            token_sequence = []
            i = 0
            j = 0
            while running_input[0, -2] != self._tokenizer.stop_idx_target and i < 200:
                prediction = self._model(context, running_input[:, :self._config["block_size"]])

                if j < self._config["block_size"]:
                    prediction = prediction[j, :]
                    next_token = torch.argmax(prediction)
                    j += 1
                else:
                    prediction = prediction[-1, :]
                    next_token = torch.argmax(prediction)

                token_sequence.append(next_token.item())

                if j < self._config["block_size"]:
                    running_input[0, j] = next_token 
                else:
                    running_input[0, -1] = next_token
                    torch.roll(running_input, -1, dims=1)
                
                i += 1
            
            print(f"Target: {batch['report_short'][0]}")
            print(f"Predic: {self._tokenizer.itos_targets(token_sequence)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of the run")
    parser.add_argument("--dataset", type=str, help="")
    parser.add_argument("--cache_data", action="store_true", help="All data will be loaded into the RAM before training")
    
    args = parser.parse_args()
    
    config = {
        "model": args.model,
        "dataset": args.dataset,
        "cached": args.cache_data,
        "block_size": 20
    }

    trainer = Generator(config)
    trainer.sample()