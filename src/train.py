import argparse
import json
import os
import tqdm
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from typing import Dict

from src.best_model import BestModel, OptDirection
from src.early_stopping import EarlyStopping, OptDirection as ESOptDirection
from src.dataloader import *
from src.loss import CELoss
from src.models import TransformerFactory
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
            
        c["src_vocab_size"] = self._context_tokenizer.size_context_vocab
        c["tgt_vocab_size"] = self._target_tokenizer.vocab_size 
        c["src_pad_idx"] = self._context_tokenizer.padding_idx_context
        c["tgt_pad_idx"] = self._target_tokenizer.padding_idx
        #c["dropout"] = 0
        
        self._model = TransformerFactory.from_dict(self._config["model"], c)

        self._model.to(DEVICE)
        self._model.save_params_to(self._config["checkpoints"])   
        print(f" [MODEL] {self._model.name}")     
        print(f" [N ELEM] {sum([param.nelement() for param in self._model.parameters()])}")

        # Loss
        self._loss = CELoss(ignore_idx=self._target_tokenizer.padding_idx)

        # Optimization
        self._optimizer = torch.optim.AdamW(self._model.parameters(), weight_decay=1e-8)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, factor=.5, patience=5)
        self._optimizer_steps = 0

        # Tensorboard
        self._writer = SummaryWriter(log_dir=self._config["tensorboard"])

        self._evaluator = Evaluator(config=config.copy())
        

    def train(self):
        best_model_by_loss = BestModel("CE_loss", OptDirection.MINIMIZE, self._config["checkpoints"])
        early_stopping_by_loss = EarlyStopping(ESOptDirection.MINIMIZE, 10, self._config["checkpoints"])

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

            if early_stopping_by_loss.update(eval_dict["loss"], epoch):
                print(f" [EARLY STOPPING] Epoch {epoch} / {self._config['epochs']}")
                break

        self._writer.close()

    def _train_epoch(self, epoch: int):
        self._model.train()

        for i, batch in enumerate(tqdm.tqdm(self._train_dataloader)):
            context = batch["overview"]
            targets = batch["report_short_wout_boeen"]

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

            # Create batch
            batch = self._batchify(context, targets, self._config["block_size"])

            for contexts, inputs, labels in zip(batch["context"], batch["inputs"], batch["labels"]):
                # Move tensors 
                #contexts = contexts.to(device=DEVICE)
                #inputs = inputs.to(device=DEVICE)
                #print(inputs.device, labels.device)
                #print(inputs.storage().data_ptr(), labels.storage().data_ptr())
                #exit()
                #labels = labels.to(device=DEVICE)

                self._optimizer.zero_grad()

                prediction = self._model(contexts, inputs)

                n_loss_values = torch.sum(torch.where(labels != self._target_tokenizer.padding_idx, 1, 0))
                labels = labels.view(labels.shape[0] * labels.shape[1])  # B * T
                
                loss = self._loss(prediction, labels) / n_loss_values
                loss.backward()

                self._optimizer.step()
            
                self._writer.add_scalar("train/loss", loss.item(), self._optimizer_steps)
                self._optimizer_steps += 1

    @staticmethod
    def _batchify(context: torch.tensor, targets: torch.tensor, block_size: int) -> Dict:
        input_seqs = []
        label_seqs = []
        context_seqs = []
        
        # Create the maximum amount of sequences from the data. Each sequence has length 'block_size' and sequences
        # are shifted by one token each.
        for j in range(0, targets.shape[1] - block_size):
            input_seqs.append(targets[:, j:j+block_size])
            label_seqs.append(targets[:, j+1:j+1+block_size])
            context_seqs.append(context)

        #print(targets[:, ::40].shape)
        # Get tensors from the list of sequences
        context = torch.concat(context_seqs)
        inputs = torch.concat(input_seqs)
        labels = torch.concat(label_seqs)

        #print(labels.shape)

        max_batch_size = 320
        n_batches = context.shape[0] // max_batch_size + 1
        actual_batch_size = context.shape[0] // n_batches
        
        # Permute sequences randomly
        perm = torch.randperm(context.shape[0])
        context_perm = context[perm, ...]
        inputs_perm = inputs[perm, ...]
        labels_perm = labels[perm, ...]

        context_perm = context_perm.to(DEVICE)
        inputs_perm = inputs_perm.to(DEVICE)
        labels_perm = labels_perm.to(DEVICE)
            
        # Create batches of size 'actual_batch_size' from the permuted tensor of sequences
        batched_contexts = []
        batched_inputs = []
        batched_labels = []
        for i in range(n_batches-1):
            batched_contexts.append(context_perm[i*actual_batch_size : (i+1)*actual_batch_size, ...])
            batched_inputs.append(inputs_perm[i*actual_batch_size : (i+1)*actual_batch_size, ...])
            batched_labels.append(labels_perm[i*actual_batch_size : (i+1)*actual_batch_size, ...])

        # If context.shape[0] is not divisible by n_batches, the last batch will have size 'actual_batch_size' + 1
        batched_contexts.append(context_perm[(n_batches-1)*actual_batch_size : , ...])
        batched_inputs.append(inputs_perm[(n_batches-1)*actual_batch_size : , ...])
        batched_labels.append(labels_perm[(n_batches-1)*actual_batch_size : , ...])

        batch = {
            "context": batched_contexts,
            "inputs": batched_inputs,
            "labels": batched_labels
        }
        
        return batch




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Name of the run")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--checkpoints_path", type=str, required=True, help="Where to store checkpoints")
    parser.add_argument("--tensorboard_path", type=str, required=True, help="Where to store tensorboard summary")
    parser.add_argument("--model", type=str, required=True, choices=["og_transformer", "rope_transformer"], help="Which model to use")
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
        "batch_size": 5,
        "epochs": 30,
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
