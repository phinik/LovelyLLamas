import argparse
import tqdm
import torch
import torch.nn.functional as F
import os

from typing import Dict

from src.dataloader import *
from src.models import TransformerFactory
from src.tokenizer import ContextTokenizer, TokenizerFactory
#import src.determinism

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
        self._context_tokenizer = ContextTokenizer(self._config["dataset"])
        self._target_tokenizer = TokenizerFactory.get(self._config["dataset"], self._config["tokenizer"], self._config["target"])

    @torch.no_grad()
    def sample(self, model):
        model.eval()

        for i, batch in enumerate(tqdm.tqdm(self._test_dataloader)):
            context = batch["overview"].copy()

            target_str = "gpt_rewritten_apokalyptisch" if self._config["target"] == "gpt" else "report_short_wout_boeen"
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
            
            print(120*'#')
            print(f"Target: {batch[target_str][0].replace('<city>', batch['city'][0])}")
            print(f"Predic: {self._target_tokenizer.itos(token_sequence).replace('<city>', batch['city'][0])}")
            for j in range(0, 192, 8):
                print(f"Overview: {batch['overview'][0].split(';')[j:j+8]}")
            print(120*'#')
            if i == 0:
                exit()

    @torch.no_grad()
    def get(self, model, id: int):
        model.eval()

        batch = get_demo_weather_dataset(self._config["dataset"])[id]
        context = batch["overview"]

        # Tokenize
        context = torch.tensor(self._context_tokenizer.stoi(context)).unsqueeze(0)
                
        context = context.to(device=DEVICE)
            
        running_input = torch.zeros(size=(1, self._config["block_size"] + 1), dtype=context.dtype).to(DEVICE)
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
            
        generated_sequence = self._target_tokenizer.itos(token_sequence)
        generated_sequence = generated_sequence.replace("<city>", batch["city"])
        generated_sequence = generated_sequence.replace(" <degC>", "°C")
        
        return generated_sequence



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of the run")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset root")
    parser.add_argument("--model_weights", type=str, help="Which model weights to use")
    parser.add_argument("--tokenizer", type=str, choices=["sow", "bert"], default="sow", help="Which tokenizer to use for the report")
    parser.add_argument("--cache_data", action="store_true", help="All data will be loaded into the RAM before training")
    parser.add_argument("--target", type=str, choices=["default", "gpt"], required=True, help="What to train on")


    args = parser.parse_args()
       
    config = {
        "name": args.name,
        "dataset": args.dataset_path,
        "model_weights": args.model_weights,
        "model_params": os.path.join(os.path.dirname(args.model_weights), "params.json"),
        "cached": args.cache_data,
        "model": "transformer", #args.model,
        "block_size": 20,
        "tokenizer": args.tokenizer,
        "target": args.target
    }

    model = TransformerFactory.from_file(config["model_params"])
    model.load_weights_from(config["model_weights"])
    model.to(DEVICE)

    generator = Generator(config)
    #generator.sample(model=model)

    print(generator.get(model, 1))