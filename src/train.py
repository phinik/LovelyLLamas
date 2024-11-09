import argparse
import json
import os
import tqdm
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from typing import Dict

#from src.best_model import BestModel, OptDirection
from src.dataloader import *
from src.loss import CELoss
from src.models import Transformer
from src.dummy_tokenizer import DummyTokenizer


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, config: Dict):
        self._config = config

        # Dataloader
        self._train_dataloader = get_train_dataloader_weather_dataset(
            path=self._config["dataset"], 
            batch_size=self._config["batch_size"],
            cached=self._config["cached"]
        )

        # Tokenizer
        self._tokenizer = DummyTokenizer(self._config["dataset"])

        # Model --> Default has 44.497.920 parameters!
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
        self._model.to(DEVICE)

        print(sum([param.nelement() for param in self._model.parameters()]))

        #exit()
        # Loss
        self._loss = CELoss(ignore_idx=self._tokenizer.padding_idx_target)

        # Optimization
        self._optimizer = torch.optim.AdamW(self._model.parameters(), weight_decay=1e-8)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, factor=.5, patience=10)

        # Tensorboard
        self._writer = SummaryWriter(log_dir=self._config["tensorboard"])
        
    def train(self):
        #best_model_by_loss = BestModel("CE_loss", OptDirection.MINIMIZE, self._config["checkpoints"])

        for epoch in range(1, self._config["epochs"] + 1):
            self._writer.add_scalar("learning_rate", self._optimizer.param_groups[0]['lr'], epoch)

            print(f" [TRAINING] Epoch {epoch}")
            self._train_epoch(epoch=epoch)

            #print(f" [EVALUATING] Epoch {epoch}")

        self._writer.close()

    def _train_epoch(self, epoch: int):
        self._model.train()

        for i, batch in enumerate(tqdm.tqdm(self._train_dataloader)):
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
            targets = targets.to(device=DEVICE)
            context = context.to(device=DEVICE)
            
            self._optimizer.zero_grad()

            total_loss = 0
            n_total_loss_values = 0
            for i in range(0, targets.shape[1] - self._config["block_size"]):
                inputs = targets[:, i:i+self._config["block_size"]]
                labels = targets[:, i+1:i+1+self._config["block_size"]]

                prediction = self._model(context, inputs)
                
                n_total_loss_values += torch.sum(torch.where(labels != self._tokenizer.padding_idx_target, 1, 0))
                labels = labels.reshape(labels.shape[0] * labels.shape[1])  # B * T
                total_loss += self._loss(prediction, labels)
                

            total_loss /= n_total_loss_values
            total_loss.backward()

            self._optimizer.step()
            
            self._writer.add_scalar("train/total_loss", total_loss.item(), epoch * (i + 1))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of the run")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset root")
    parser.add_argument("--checkpoints_path", type=str, help="Where to store checkpoints")
    parser.add_argument("--tensorboard_path", type=str, help="Where to store tensorboard summary")
    parser.add_argument("--model", type=str, choices=["transformer", "lstm"], help="Which model to use")
    parser.add_argument("--cache_data", action="store_true", help="All data will be loaded into the RAM before training")
    
    args = parser.parse_args()
    
    config = {
        "dataset": args.dataset_path,
        "checkpoints": os.path.join(args.checkpoints_path, args.name),
        "tensorboard": os.path.join(args.tensorboard_path, args.name),
        "cached": args.cache_data,
        "model": args.model,
        "batch_size": 5,
        "epochs": 10,
        "block_size": 20
    }

    os.makedirs(config["checkpoints"], exist_ok=True)
    os.makedirs(config["tensorboard"], exist_ok=True)

    with open(os.path.join(config["checkpoints"], "config.json"), "w") as f:
        json.dump(config, f, sort_keys=True, indent=4)

    print(f" [DEVICE] Using {DEVICE}")
    trainer = Trainer(config)
    trainer.train()