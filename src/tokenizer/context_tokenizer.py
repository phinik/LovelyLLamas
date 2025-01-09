import json
import os

from typing import List


class ContextTokenizer:
    SPECIAL_TOKENS = ["<unknown>", "<padding>", "<missing>"]

    def __init__(self, dataset_path: str):
        super().__init__()

        self._dataset_path = dataset_path

        self._context_tokens = self._load_tokens(os.path.join(self._dataset_path, "context_tokens.json"))
        self._context_tokens += self.SPECIAL_TOKENS
        self._context_tokens = sorted(list(set(self._context_tokens)))

        self._stoi = {token: i for i, token in enumerate(self._context_tokens)}

    def _load_tokens(self, filename: str) -> List:
        with open(os.path.join(self._dataset_path, filename), "r") as f:
            return json.load(f)
    
    @property
    def padding_idx(self) -> int:
        return self._stoi["<padding>"]
    
    @property
    def unknown_idx(self) -> int:
        return self._stoi["<unknown>"]
    
    @property
    def missing_idx(self) -> int:
        return self._stoi["<missing>"]
    
    @property
    def vocab_size(self) -> int:
        return len(self._stoi)
    
    @property
    def vocab_size(self) -> int:
        return max(self._stoi.values()) + 1
    
    def stoi(self, input: str) -> List[int]:        
        words = input.split(";")

        return [self._stoi.get(x, self._stoi["<unknown>"]) for x in words]
