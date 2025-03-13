import argparse
import tqdm
import os
import json

from typing import Set

from src.dataset import *
from src.dataloader import *
from src.tokenizer import BertTokenizerAdapter


def process(dataloader) -> Set:
    tokens_target_total = set()

    tokenizer = BertTokenizerAdapter()

    for item in tqdm.tqdm(dataloader):
        target = item["report_short_wout_boeen"][0]
        
        token_seq = tokenizer.stoi(target)
        
        tokens_target_total = tokens_target_total.union(token_seq)
        
    numbers = " ".join([str(i) for i in range(-60, 61)])
    numbers = tokenizer.stoi(numbers)
    tokens_target_total = tokens_target_total.union(numbers)

    return tokens_target_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset", required=True)
    parser.add_argument("--output_directory", type=str, help="Where to store the sets", required=True)

    args = parser.parse_args()

    print("[TRAIN]")
    train_dataloader = get_train_dataloader_weather_dataset(
        path=args.dataset, 
        batch_size=1,
        num_workers=1, 
        cached=False
    )
    tokens_target_total_train = process(train_dataloader)

    print("[EVAL]")
    eval_dataloader = get_eval_dataloader_weather_dataset(
        path=args.dataset, 
        batch_size=1,
        num_workers=1, 
        cached=False
    )
    tokens_target_total_eval = process(eval_dataloader)

    print("[TEST]")
    test_dataloader = get_test_dataloader_weather_dataset(
        path=args.dataset, 
        batch_size=1,
        cached=False
    )
    tokens_target_total_test = process(test_dataloader)

    tokens_target_total = tokens_target_total_train.union(tokens_target_total_eval).union(tokens_target_total_test)

    print(len(tokens_target_total))

    with open(os.path.join(args.output_directory, "rgpt_tokens_bert.json"), "wb") as f:
        tokens_jsonized = json.dumps(sorted(list(tokens_target_total)), ensure_ascii=False, indent=4).encode("utf-8")
        f.write(tokens_jsonized)
