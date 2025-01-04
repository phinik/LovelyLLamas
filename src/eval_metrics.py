import argparse
import json
import tqdm
import torch
import torch.nn.functional as F
import os

from typing import Dict

from src.dataloader import *
from src.models import TransformerFactory
from src.dummy_tokenizer import DummyTokenizer
from src.tokenizer import Tokenizer
from src.metrics import IMetric, BertScore, Bleu, Rouge
import src.determinism

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self, config: Dict, metrics: List[IMetric]):
        self._config = config

        # Dataloader
        self._test_dataloader = get_eval_dataloader_weather_dataset(
            path=self._config["dataset"], 
            batch_size=1,
            num_workers=self._config["num_workers"],
            cached=self._config["cached"]
        )

        # Tokenizer
        self._context_tokenizer = DummyTokenizer(self._config["dataset"])
        self._target_tokenizer = DummyTokenizer(self._config["dataset"]) if self._config["tokenizer"] == "dummy" else Tokenizer()
        
        self._metrics = metrics

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()

        for i, batch in enumerate(tqdm.tqdm(self._test_dataloader)):
            context = batch["overview"].copy()
            targets = batch["report_short_wout_boeen"].copy()

            # Tokenize
            for j in range(len(context)):
                context[j] = torch.tensor(self._context_tokenizer.stoi_context(context[j])).unsqueeze(0)
                targets[j] = torch.tensor(self._target_tokenizer.stoi("<start> " + targets[j] + " <stop>"))

            context = context[0]
            targets = targets[0]

            # Move tensors 
            targets = targets.to(device=DEVICE)
            context = context.to(device=DEVICE)
            
            running_input = torch.zeros(size=(1, self._config["block_size"] + 1), dtype=targets.dtype).to(DEVICE)
            running_input[0, 0] = self._target_tokenizer.start_idx
            token_sequence = []

            j = 0
            k = 0
            while running_input[0, -2] != self._target_tokenizer.stop_idx and k < 200:
                prediction = model(context, running_input[:, :self._config["block_size"]])

                if j < self._config["block_size"]:
                    prediction = prediction[j, :]
                    next_token = torch.multinomial(F.softmax(prediction, dim=0), 1)
                    j += 1
                else:
                    prediction = prediction[-1, :]
                    next_token = torch.multinomial(F.softmax(prediction, dim=0), 1)

                token_sequence.append(next_token.item())

                if j < self._config["block_size"]:
                    running_input[0, j] = next_token 
                else:
                    running_input[0, -1] = next_token
                    running_input = torch.roll(running_input, -1, dims=1)
                
                k += 1
            
            for metric in self._metrics:
                metric.update(self._target_tokenizer.itos(token_sequence), batch['report_short_wout_boeen'][0])

        results = {}
        for metric in self._metrics:
            results[metric.name] = metric.get()

        return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of the run")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset root")
    parser.add_argument("--model_weights", type=str, help="Which model weights to use")
    parser.add_argument("--cache_data", action="store_true", help="All data will be loaded into the RAM before training")
    parser.add_argument("--tokenizer", type=str, choices=["dummy", "bert"], default="dummy", help="Which tokenizer to use for the report")
    parser.add_argument("--metrics", nargs="+", choices=["bertscore", "bleu", "rouge"], type=str, help="", required=True)
    parser.add_argument("--output_filename", type=str, help="If output shall be saved to a different file than the standard file")
    
    args = parser.parse_args()
    
    config = {
        "name": args.name,
        "dataset": args.dataset_path,
        "model_weights": args.model_weights,
        "model_params": os.path.join(os.path.dirname(args.model_weights), "params.json"),
        "cached": args.cache_data,
        "block_size": 20,
        "tokenizer": args.tokenizer,
        "num_workers": 1
    }

    model = TransformerFactory.from_dict(config["model_params"])
    model.load_weights_from(config["model_weights"])
    model.to(DEVICE)

    metrics = []
    for metric in args.metrics:
        if metric == "bertscore":
            metrics.append(BertScore())
        if metric == "bleu":
            metrics.append(Bleu())
        if metric == "rouge":
            metrics.append(Rouge())

    generator = Evaluator(config, metrics=metrics)
    results = generator.evaluate(model)

    out_dir = os.path.dirname(config["model_weights"])
    
    if args.output_filename is not None:
        filename = f"{args.output_filename}.json"
    else:
        filename = f"eval_{os.path.splitext(os.path.split(config['model_weights'])[1])[0]}.json"
    with open(os.path.join(out_dir, filename), "w") as f:
        json.dump(results, f, indent=4)