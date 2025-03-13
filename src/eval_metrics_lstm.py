import argparse
import json
import tqdm
import torch
import torch.nn.functional as F
import os

from typing import Dict

from src.dataloader import *
from src.models.lstm import LSTM
from src.tokenizer import ContextTokenizer, TokenizerFactory
from src.metrics import IMetric, BertScore, Bleu, Rouge, CityAppearance, TemperatureCorrectness, CustomClassifier, TemperatureRange, CustomClassifierCT
from src.data_postprocessing import PostProcess
from src.utils import OverviewSelector, TargetSelector


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
        self._context_tokenizer = ContextTokenizer(self._config["dataset"])
        self._target_tokenizer = TokenizerFactory.get(
            self._config["dataset"], 
            self._config["tokenizer"], 
            self._config["target"]
        )
        
        self._metrics = metrics

        self._post_processor = PostProcess()

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()

        target_str = TargetSelector.select(self._config["target"])
        overview_str = OverviewSelector.select(self._config["overview"])

        print(f" [TARGET] {target_str.upper()}")
        print(f" [OVERVIEW] {overview_str.upper()}")

        for i, batch in enumerate(tqdm.tqdm(self._test_dataloader)):
            context = batch[overview_str].copy()
            targets = batch[target_str].copy()

            # Tokenize
            for j in range(len(context)):
                context[j] = torch.tensor(self._context_tokenizer.stoi(context[j])).unsqueeze(0)
                targets[j] = torch.tensor(self._target_tokenizer.stoi(self._target_tokenizer.add_start_stop_tokens(targets[j])))

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
            
            # Remove start token if present
            if token_sequence[0] == self._target_tokenizer.start_idx:
                token_sequence = token_sequence[1:]

            # Remove stop token if present
            if token_sequence[-1] == self._target_tokenizer.stop_idx:
                token_sequence = token_sequence[:-1]

            for metric in self._metrics:
                metric.update(
                    prediction=self._post_processor(self._target_tokenizer.itos(token_sequence)), 
                    tokenized_prediction=token_sequence,
                    label=self._post_processor(batch[target_str][0]),  # post processing only to be on the safe side
                    contexts={
                        "overview_full": batch["overview_full"][0],
                        "overview_ct": batch["overview_ct"][0]
                    },     
                    temperature=[t[0] for t in batch["temperatur_in_deg_C"]]
                )
            
        results = {}
        for metric in self._metrics:
            results[metric.name] = metric.get()

        return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        help="Path to dataset root"
    )
    
    parser.add_argument(
        "--model_weights", 
        type=str, 
        help="Which model weights to use"
    )
        
    parser.add_argument(
        "--metrics", 
        nargs="+", 
        choices=["bertscore", "bleu", "rouge", "temps", "temp_range", "cities", "classifier", "classifier_ct"], 
        type=str, 
        help="Select which metrics shall be computed. Note: 'classifier' and 'classifier_ct' do not work with SoW models", 
        required=True
    )
    parser.add_argument(
        "--output_filename", 
        type=str, 
        help="If output shall be saved to a different file than the standard file"
    )
    
    args = parser.parse_args()
    
    
    config_path = os.path.join(os.path.dirname(args.model_weights), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
   
    config["dataset"] = args.dataset_path
    config["model_weights"] = args.model_weights
    config["model_params"] = os.path.join(os.path.dirname(args.model_weights), "params.json")
    config["num_workers"] = 1
    if "overview" not in config.keys():
        config["overview"] = "full"
    
    with open(config["model_params"], "r") as f:
        params = json.load(f)

    c = {k: v for k, v in params.items() if k in LSTM.__init__.__code__.co_varnames}

    model = LSTM(**c)
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
        if metric == "temps":
            metrics.append(TemperatureCorrectness())
        if metric == "cities":
            metrics.append(CityAppearance())
        if metric == "temp_range":
            metrics.append(TemperatureRange())
        if metric == "classifier":
            metrics.append(CustomClassifier(dataset_path=config["dataset"]))
        if metric == "classifier_ct":
            metrics.append(CustomClassifierCT(dataset_path=config["dataset"]))
        
    generator = Evaluator(config, metrics=metrics)
    results = generator.evaluate(model)

    out_dir = os.path.dirname(config["model_weights"])
    
    if args.output_filename is not None:
        filename = f"{args.output_filename}.json"
    else:
        filename = f"eval_{os.path.splitext(os.path.split(config['model_weights'])[1])[0]}.json"
    with open(os.path.join(out_dir, filename), "w") as f:
        json.dump(results, f, indent=4)

    print(results)