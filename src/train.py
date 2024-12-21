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
from src.tokenizer import Tokenizer
from src.eval import Evaluator
import src.determinism


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, config: Dict):
        self._config = config

        # Dataloader
        self._train_dataloader = get_train_dataloader_weather_dataset(
            path=self._config["dataset"], 
            batch_size=self._config["batch_size"],
            num_workers=self._config["num_workers"],
            cached=self._config["cached"]
        )

        # Tokenizer
        self._context_tokenizer = DummyTokenizer(self._config["dataset"])
        self._target_tokenizer = DummyTokenizer(self._config["dataset"]) if self._config["tokenizer"] == "dummy" else Tokenizer()

        with open(self._config["model_config"], "r") as f:
            c = json.load(f)
            
        c["n_src_vocab"] = self._context_tokenizer.size_context_vocab
        c["n_trg_vocab"] = self._target_tokenizer.vocab_size 
        c["src_pad_idx"] = self._context_tokenizer.padding_idx_context
        c["trg_pad_idx"] = self._target_tokenizer.padding_idx
        
        self._model = Transformer(**c)

        self._model.to(DEVICE)
        self._model.save_params_to(self._config["checkpoints"])        
        print(f" [N ELEM] {sum([param.nelement() for param in self._model.parameters()])}")

        #exit()
        # Loss
        self._loss = CELoss(ignore_idx=self._target_tokenizer.padding_idx)

        # Optimization
        self._optimizer = torch.optim.AdamW(self._model.parameters(), weight_decay=1e-8)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, factor=.5, patience=10)  # TODO Patience auf 5 reduzieren

        # Tensorboard
        self._writer = SummaryWriter(log_dir=self._config["tensorboard"])

        self._evaluator = Evaluator(config=config.copy())
        

    def train(self):
        best_model_by_loss = BestModel("CE_loss", OptDirection.MINIMIZE, self._config["checkpoints"])

        for epoch in range(1, self._config["epochs"] + 1):
            self._writer.add_scalar("learning_rate", self._optimizer.param_groups[0]['lr'], epoch)

            print(f" [TRAINING] Epoch {epoch} / {self._config['epochs']}")
            self._train_epoch(epoch=epoch)

            print(f" [EVALUATING] Epoch {epoch} / {self._config['epochs']}")
            eval_dict = self._evaluator.evaluate(self._model)
            print(f"              Loss: {eval_dict['loss']}")

            self._writer.add_scalar("eval/loss", eval_dict["loss"], epoch)

            self._scheduler.step(eval_dict["loss"])

            self._model.save_weights_as(self._config["checkpoints"], f"checkpoint_epoch_{epoch}")

            best_model_by_loss.update(epoch, self._model, eval_dict["loss"])

        self._writer.close()

    def _train_epoch(self, epoch: int):
        self._model.train()

        for i, batch in enumerate(tqdm.tqdm(self._train_dataloader)):
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
            
            self._optimizer.zero_grad()

            total_loss = 0
            n_total_loss_values = 0
            for j in range(0, targets.shape[1] - self._config["block_size"]):
                inputs = targets[:, j:j+self._config["block_size"]]
                labels = targets[:, j+1:j+1+self._config["block_size"]]

                prediction = self._model(context, inputs)
                
                n_total_loss_values += torch.sum(torch.where(labels != self._target_tokenizer.padding_idx, 1, 0))
                labels = labels.reshape(labels.shape[0] * labels.shape[1])  # B * T
                total_loss += self._loss(prediction, labels)
                

            total_loss /= n_total_loss_values
            total_loss.backward()

            self._optimizer.step()
            
            self._writer.add_scalar("train/loss", total_loss.item(), (epoch-1) * len(self._train_dataloader) + i)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of the run")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset root")
    parser.add_argument("--checkpoints_path", type=str, help="Where to store checkpoints")
    parser.add_argument("--tensorboard_path", type=str, help="Where to store tensorboard summary")
    parser.add_argument("--model", type=str, choices=["transformer", "lstm"], help="Which model to use")
    parser.add_argument("--cache_data", action="store_true", help="All data will be loaded into the RAM before training")
    parser.add_argument("--tokenizer", type=str, choices=["dummy", "bert"], default="dummy", help="Which tokenizer to use for the report")
    parser.add_argument("--model_config", type=str, required=True, help="What transformer model configuration to use")
    parser.add_argument("--num_workers", type=int, default=4, help="How many workers to use for dataloading")

    
    args = parser.parse_args()
    
    config = {
        "name": args.name,
        "dataset": args.dataset_path,
        "checkpoints": os.path.join(args.checkpoints_path, args.name),
        "tensorboard": os.path.join(args.tensorboard_path, args.name),
        "cached": args.cache_data,
        "model": args.model,
        "batch_size": 10,
        "epochs": 40,
        "block_size": 20,
        "tokenizer": args.tokenizer,
        "model_config": args.model_config,
        "num_workers": args.num_workers
    }

    os.makedirs(config["checkpoints"], exist_ok=True)
    os.makedirs(config["tensorboard"], exist_ok=True)

    with open(os.path.join(config["checkpoints"], "config.json"), "w") as f:
        json.dump(config, f, sort_keys=True, indent=4)

    print(f" [DEVICE] {DEVICE}")
    trainer = Trainer(config)
    trainer.train()
