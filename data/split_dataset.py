import argparse
import os
import random
import shutil

from typing import List, Dict


def load_filenames(path: str) -> List[str]:
    files = os.listdir(path)
    
    # Only load filenames with suffix "standardised"
    return [f for f in files if f.endswith("_standardised.json")]

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
        dest_path = os.path.join(split_directory, file)

        shutil.copyfile(src_path, dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_directory", type=str, help="Path to directory containing the json files")
    parser.add_argument("destination_directory", type=str, help="Path to directory where the splits will be stored")
    parser.add_argument("--train_split", type=float, default=0.6, help="Wanted fraction of data for training")
    parser.add_argument("--eval_split", type=float, default=0.3, help="Wanted fraction of data for evaluation")

    args = parser.parse_args()

    assert os.path.exists(args.source_directory), "'source_directory' does not exist"
    assert os.path.exists(args.destination_directory), "'destination_directory' does not exist"
    assert args.train_split + args.eval_split < 1.0, "'--train_split' and '--eval_split' cannot sum to 1 or greater"

    filenames = load_filenames(args.source_directory)

    splits = create_splits(
        filenames=filenames,
        train_split=args.train_split,
        eval_split=args.eval_split
    )

    print(f" The following splits will be created:")
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
