import argparse
import json
import tqdm
import torch
import os

from typing import Dict

from src.dataloader import *
from src.models import Transformer
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
            cached=self._config["cached"]
        )

        # Tokenizer
        self._context_tokenizer = DummyTokenizer(self._config["dataset"])
        self._target_tokenizer = DummyTokenizer(self._config["dataset"])#Tokenizer()
        
        self._metrics = metrics

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()

        for i, batch in enumerate(tqdm.tqdm(self._test_dataloader)):
            context = batch["overview"].copy()
            targets = batch["report_short"].copy()

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
                    next_token = torch.multinomial(torch.nn.functional.softmax(prediction, dim=0), 1) #torch.argmax(prediction)
                    j += 1
                else:
                    prediction = prediction[-1, :]
                    next_token = torch.multinomial(torch.nn.functional.softmax(prediction, dim=0), 1) #torch.argmax(prediction)

                token_sequence.append(next_token.item())

                if j < self._config["block_size"]:
                    running_input[0, j] = next_token 
                else:
                    running_input[0, -1] = next_token
                    running_input = torch.roll(running_input, -1, dims=1)
                
                k += 1
            
            for metric in self._metrics:
                metric.update(self._target_tokenizer.itos(token_sequence), batch['report_short'][0])

        results = {}
        for metric in self._metrics:
            results[metric.name] = metric.get()

        return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of the run")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset root")
    parser.add_argument("--model_weights", type=str, help="Which model weights to use")
    #parser.add_argument("--model_params", type=str, help="Which model params to use")
    #parser.add_argument("--model", type=str, choices=["transformer", "lstm"], help="Which model to use")
    parser.add_argument("--cache_data", action="store_true", help="All data will be loaded into the RAM before training")
    parser.add_argument("--metrics", nargs="+", choices=["bertscore", "bleu", "rouge"], type=str, help="", required=True)
    
    args = parser.parse_args()
    
    config = {
        "name": args.name,
        "dataset": args.dataset_path,
        "model_weights": args.model_weights,
        "model_params": os.path.join(os.path.dirname(args.model_weights), "params.json"),
        "cached": args.cache_data,
        "model": "transformer", #args.model,
        "block_size": 20
    }

    model = Transformer.from_params(config["model_params"])
    model.load_weights_from(config["model_weights"])
    model.to(DEVICE)

    metrics = []
    for metric in args.metrics:
        if metric == "bertscore":
            metrics.append(BertScore(config["dataset"]))
        if metric == "bleu":
            metrics.append(Bleu())
        if metric == "rouge":
            metrics.append(Rouge())

    generator = Evaluator(config, metrics=metrics)
    results = generator.evaluate(model)

    out_dir = os.path.dirname(config["model_weights"])
    filename = f"eval_{os.path.splitext(os.path.split(config['model_weights'])[1])[0]}.json"
    with open(os.path.join(out_dir, filename), "w") as f:
        json.dump(results, f, indent=4)