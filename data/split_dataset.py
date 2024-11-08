import argparse
import json
import os
import random
import shutil

from typing import List, Dict

def contains_data(path: str, filename: str) -> bool:
    with open(os.path.join(path, filename), 'r') as f:
        data = json.load(f)

    required_keys = ["city", "created_day", "report_short", "report_long", "overview"]

    for key in required_keys:
        if data[key] is None:
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

def create_splits(filenames: List[str], train_split: float, eval_split: float) -> Dict:
    random.shuffle(filenames)

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

def copy_files(source_directory: str, destination_directory: str, key: str, files: List[str]):
    split_directory = os.path.join(destination_directory, key)

    # Create split directory
    os.makedirs(split_directory)

    # Copy files to split directory
    for file in files:
        src_path = os.path.join(source_directory, file)
        dest_path = os.path.join(split_directory, file.replace(os.sep, "_"))

        shutil.copyfile(src_path, dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_directory", type=str, help="Path to directory containing the json files")
    parser.add_argument("destination_directory", type=str, help="Path to directory where the splits will be stored")
    parser.add_argument("--train_split", type=float, default=0.7, help="Wanted fraction of data for training")
    parser.add_argument("--eval_split", type=float, default=0.2, help="Wanted fraction of data for evaluation")

    args = parser.parse_args()

    assert os.path.exists(args.source_directory), "'source_directory' does not exist"
    assert os.path.exists(args.destination_directory), "'destination_directory' does not exist"
    assert args.train_split + args.eval_split < 1.0, "'--train_split' and '--eval_split' cannot sum to 1 or greater"

    print(f" Searching for files:")
    filenames = load_filenames(args.source_directory)
    
    splits = create_splits(
        filenames=filenames,
        train_split=args.train_split,
        eval_split=args.eval_split
    )

    print(80*'-')
    print(f" The following splits will be created using {len(filenames)} files:")
    print(f"     Train: {len(splits['train'])} files")
    print(f"     Eval:  {len(splits['eval'])} files")
    print(f"     Test:  {len(splits['test'])} files")

    for key, value in splits.items():
        copy_files(
            source_directory=args.source_directory,
            destination_directory=args.destination_directory,
            key=key,
            files=value
        )

    print(80*'-')
    print(" Dataset successfully created")

