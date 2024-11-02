import argparse

from models import Transformer
from dataloader import *

class Trainer:
    def __init__(self):
        pass

    def train(self):
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="")
    parser.add_argument("--checkpoint_path", type=str, help="")
    parser.add_argument("--tensorboard_path", type=str, help="")
    parser.add_argument("--model", type=str, choices=["transformer", "lstm"], help="")
    
    args = parser.parse_args()
    
    trainer = Trainer()
    trainer.train()