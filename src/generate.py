import argparse
import tqdm
import torch
import torch.nn.functional as F
import os

from typing import Dict

from src.dataloader import *
from src.models import Transformer
from src.models import LSTM
from src.dummy_tokenizer import DummyTokenizer
from src.tokenizer import Tokenizer


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
        self._context_tokenizer = DummyTokenizer(self._config["dataset"])
        self._target_tokenizer = DummyTokenizer(self._config["dataset"]) if self._config["tokenizer"] == "dummy" else Tokenizer()

    @torch.no_grad()
    @torch.no_grad()
    def sample(self, model):
        model.eval()

        for i, batch in enumerate(tqdm.tqdm(self._test_dataloader)):
            context = batch["overview"].copy()
            context = batch["overview"].copy()
            targets = batch["report_short"].copy()

            # Tokenize
            for j in range(len(context)):
                context[j] = torch.tensor(self._tokenizer.stoi_context(context[j])).unsqueeze(0)
                targets[j] = torch.tensor(self._tokenizer.stoi_targets("<start> " + targets[j] + " <stop>"))

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
            while running_input[0, -2] != self._tokenizer.stop_idx_target and k < 200:
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
                k += 1
            
            print(" ")
            print(f"Target: {batch['report_short'][0]}")
            print(f"Overview: {batch['overview'][0]}")
            print(f"Predic: {self._tokenizer.itos_targets(token_sequence).replace('<city>', batch['city'][0])}")

            exit()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of the run")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset root")
    parser.add_argument("--model_weights", type=str, help="Which model weights to use")
    #parser.add_argument("--model", type=str, choices=["transformer", "lstm"], help="Which model to use")
    parser.add_argument("--tokenizer", type=str, choices=["dummy", "bert"], default="dummy", help="Which tokenizer to use for the report")
    parser.add_argument("--cache_data", action="store_true", help="All data will be loaded into the RAM before training")

    args = parser.parse_args()
       
    config = {
        "name": args.name,
        "dataset": args.dataset_path,
        "model_weights": args.model_weights,
        "model_params": os.path.join(os.path.dirname(args.model_weights), "params.json"),
        "cached": args.cache_data,
        "model": "LSTM", #args.model,
        "block_size": 20
    }

    model = LSTM.from_params(config["model_params"])
    model.load_weights_from(config["model_weights"])
    model.to(DEVICE)

    generator = Generator(config)
    generator.sample(model)