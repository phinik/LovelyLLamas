import json
import os

from typing import List


class ContextTokenizer:
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path

        self._vocab = self._load_vocab(os.path.join(self._dataset_path, "context_vocab.json"))
        
        # The transformer requires a padding_idx for the encoder, even though we never use it...
        self._vocab += ["<padding>"]  

        self._stoi = {word: i for i, word in enumerate(self._vocab)}

    def _load_vocab(self, filename: str) -> List:
        with open(os.path.join(self._dataset_path, filename), "r") as f:
            return json.load(f)

    @property
    def padding_idx(self) -> int:
        return self._stoi["<padding>"]
        
    @property
    def vocab_size(self) -> int:
        return max(self._stoi.values()) + 1
        
    def stoi(self, input: str) -> List[int]:        
        words = input.split(",")

        return [self._stoi[x] for x in words]
