import argparse
import tqdm
import torch
import torch.nn as nn
import os

from typing import Dict
from src.dataloader import get_eval_dataloader_weather_dataset_classifier
from src.tokenizer import TokenizerFactory, ContextTokenizer
from src.loss import BCELoss
from src.models import TransformerFactory
import src.determinism
from src import utils


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EvaluatorClassifier:
    def __init__(self, config: Dict):
        self._config = config

        # Dataloader
        self._eval_dataloader = get_eval_dataloader_weather_dataset_classifier(
            path=self._config["dataset"], 
            batch_size=self._config["batch_size"],
            num_workers=self._config["num_workers"],
            overview=self._config["overview"]
        )

        # Tokenizer
        self._context_tokenizer = ContextTokenizer(self._config["dataset"])
        self._target_tokenizer = TokenizerFactory.get(self._config["dataset"], self._config["tokenizer"], self._config["target"])

        # Loss
        self._loss = BCELoss()

    @torch.no_grad()
    def evaluate(self, model) -> Dict:
        model.eval()

        total_loss = 0
        total_loss_values = 0

        for i, batch in enumerate(tqdm.tqdm(self._eval_dataloader)):
            context = batch["overview"]
            targets_class_0 = batch["class_0"]
            targets_class_1 = batch["class_1"]

            # Tokenize
            for j in range(len(context)):
                context[j] = torch.tensor(self._context_tokenizer.stoi(context[j])).unsqueeze(0)
                targets_class_0[j] = torch.tensor(self._target_tokenizer.stoi(targets_class_0[j]))
                targets_class_1[j] = torch.tensor(self._target_tokenizer.stoi(targets_class_1[j]))

            # Pad target sequences to have equal length and transform the list of tensors into a single tensor.
            targets_class_0 = nn.utils.rnn.pad_sequence(
                targets_class_0, 
                padding_value=self._target_tokenizer.padding_idx, 
                batch_first=True
            )

            targets_class_1 = nn.utils.rnn.pad_sequence(
                targets_class_1, 
                padding_value=self._target_tokenizer.padding_idx, 
                batch_first=True
            )

            # Context sequences always have equal length, hence, no padding is required and the list of tensors is just
            # concatenated.
            context = torch.concat(context)

            # Create batch
            batch = utils.batchify_classifier(
                context=context, 
                targets_class_0=targets_class_0, 
                targets_class_1=targets_class_1, 
                block_size=self._config["block_size"], 
                pad_idx=self._target_tokenizer.padding_idx, 
                device=DEVICE
            )
            
            for contexts, inputs, labels in zip(batch["context"], batch["inputs"], batch["labels"]):    
                prediction = model(contexts, inputs)
                prediction = nn.functional.sigmoid(prediction)

                total_loss_values += labels.shape[0]
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

    model = TransformerFactory.from_file(config["model_params"])
    model.load_weights_from(config["model_weights"])
    model.to(DEVICE)

    evaluator = EvaluatorClassifier(config)
    res = evaluator.evaluate(model)

    print(res)
