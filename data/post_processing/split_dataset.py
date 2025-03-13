"""
This file was used to create the dataset splits.
"""
import argparse
import json
import os
import random
import shutil

from typing import List, Dict

def contains_data(path: str, filename: str) -> bool:
    with open(os.path.join(path, filename), 'r') as f:
        data = json.load(f)

    required_keys = ["city", "created_day", "times", "report_short_wout_boeen"] #"gpt_rewritten_cleaned"

    for key in required_keys:
        if key not in data.keys():
            return False
        
        if data[key] is None:
            return False
        
    if len(data["times"]) != 24:
        return False
        
    return True

def load_filenames(path: str) -> List[str]:
    days = os.listdir(path)

    filenames_all = []
    for day in days:
        day_path = os.path.join(path, day)
        files = os.listdir(day_path)
        
        # Only load filenames with suffix "standardised"
        filenames = [f for f in files if f.endswith("_standardised.json")]

        # Only load filename that have data for certain keys
        filenames_with_data = [f for f in filenames if contains_data(day_path, f)]

        print(f"     Found {len(filenames)} files in {day_path} - {len(filenames_with_data)} contain data")

        filenames_all += [os.path.join(day, f) for f in filenames_with_data]

    return filenames_all

def create_splits(filenames: List[str], train_split: float, eval_split: float, size: int) -> Dict:
    random.shuffle(filenames)

    # Constrain the size of the dataset to be at most "size"
    if size != -1 and len(filenames) > size:
        filenames = filenames[:size]

    # Determine number of files per split
    n_train = int(len(filenames) * train_split)
    n_eval = int(len(filenames) * eval_split)
    n_test = len(filenames) - n_train - n_eval

    # Create splits
    train_files = filenames[:n_train]
    eval_files = filenames[n_train:(n_train + n_eval)]
    test_files = filenames[(n_train + n_eval):]

    return {
        "train": train_files,
        "eval": eval_files,
        "test": test_files
    }

def copy_files(source_directory: str, destination_directory: str, files: List[str]):
    data_directory = os.path.join(destination_directory, "data")

    # Create split directory
    os.makedirs(data_directory, exist_ok=True)

    # Copy files to split directory
    for file in files:
        os.makedirs(os.path.join(data_directory, os.path.dirname(file)), exist_ok=True)

        src_path = os.path.join(source_directory, file)
        dest_path = os.path.join(data_directory, file)

        shutil.copyfile(src_path, dest_path)

def save_file_to_disk(destination_directory: str, split: str, files: List[str]):
    full_path_files = [os.path.join("data", f) for f in files]

    with open(os.path.join(destination_directory, f"dset_{split}.json"), 'w') as f:
        json.dump(full_path_files, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_directory", type=str, help="Path to directory containing the json files")
    parser.add_argument("destination_directory", type=str, help="Path to directory where the splits will be stored")
    parser.add_argument("--train_split", type=float, default=0.7, help="Wanted fraction of data for training")
    parser.add_argument("--eval_split", type=float, default=0.2, help="Wanted fraction of data for evaluation")
    parser.add_argument("--size", type=int, default=-1, help="Number of files to consider for creating the dataset. If -1, all files will be considered.")

    args = parser.parse_args()

    assert os.path.exists(args.source_directory), "'source_directory' does not exist"
    assert os.path.exists(args.destination_directory), "'destination_directory' does not exist"
    assert args.train_split + args.eval_split < 1.0, "'--train_split' and '--eval_split' cannot sum to 1 or greater"

    print(f" Searching for files:")
    filenames = load_filenames(args.source_directory)

    splits = create_splits(
        filenames=filenames,
        train_split=args.train_split,
        eval_split=args.eval_split,
        size=args.size
    )

    print(80*'-')
    print(f" The following splits will be created using {len(filenames)} files:")
    print(f"     Train: {len(splits['train'])} files")
    print(f"     Eval:  {len(splits['eval'])} files")
    print(f"     Test:  {len(splits['test'])} files")

    copy_files(
        source_directory=args.source_directory,
        destination_directory=args.destination_directory,
        files=splits["test"] + splits["eval"] + splits["train"]
    )

    for key, value in splits.items():
        save_file_to_disk(
            destination_directory=args.destination_directory,
            split=key,
            files=value
        )

    print(80*'-')
    print(" Dataset successfully created")

