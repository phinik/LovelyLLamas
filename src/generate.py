import argparse
import tqdm
import torch

from typing import Dict

from src.dataloader import *
from src.models import Transformer
from src.dummy_tokenizer import DummyTokenizer


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

    def sample(self, model):
        model.eval()

        for i, batch in enumerate(tqdm.tqdm(self._test_dataloader)):
            context = batch["overview"]
            targets = batch["report_short"].copy()

            # Tokenize
            for i in range(len(context)):
                context[i] = torch.tensor(self._tokenizer.stoi_context(context[i])).unsqueeze(0)
                targets[i] = torch.tensor(self._tokenizer.stoi_targets("<start> " + targets[i] + " <stop>"))

            context = context[0]
            targets = targets[0]

            # Move tensors 
            targets = targets.to(device=DEVICE)
            context = context.to(device=DEVICE)
            
            running_input = torch.zeros(size=(1, self._config["block_size"] + 1), dtype=targets.dtype).to(DEVICE)
            running_input[0, 0] = self._tokenizer.start_idx_target
            token_sequence = []
            i = 0
            j = 0
            while running_input[0, -2] != self._tokenizer.stop_idx_target and i < 200:
                prediction = model(context, running_input[:, :self._config["block_size"]])

                if j < self._config["block_size"]:
                    prediction = prediction[j, :]
                    next_token = torch.multinomial(torch.softmax(prediction), 1) #torch.argmax(prediction)
                    j += 1
                else:
                    prediction = prediction[-1, :]
                    next_token = torch.multinomial(torch.softmax(prediction), 1) #torch.argmax(prediction)

                token_sequence.append(next_token.item())

                if j < self._config["block_size"]:
                    running_input[0, j] = next_token 
                else:
                    running_input[0, -1] = next_token
                    running_input = torch.roll(running_input, -1, dims=1)
                
                i += 1
            
            print(f"Target: {batch['report_short'][0]}")
            print(f"Predic: {self._tokenizer.itos_targets(token_sequence)}")

            exit()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of the run")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset root")
    parser.add_argument("--model_weights", type=str, help="Which model weights to use")
    parser.add_argument("--model_params", type=str, help="Which model params to use")
    #parser.add_argument("--model", type=str, choices=["transformer", "lstm"], help="Which model to use")
    parser.add_argument("--cache_data", action="store_true", help="All data will be loaded into the RAM before training")

    args = parser.parse_args()
    
    config = {
        "name": args.name,
        "dataset": args.dataset_path,
        "model_weights": args.model_weights,
        "model_params": args.model_params,
        "cached": args.cache_data,
        "model": "transformer", #args.model,
        "block_size": 20
    }

    model = Transformer.from_params(config["model_params"])
    model.load_weights_from(config["model_weights"])
    model.to(DEVICE)

    generator = Generator(config)
    generator.sample(model)