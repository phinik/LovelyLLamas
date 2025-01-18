import json
import os

from typing import List


class DummyTokenizer:
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path
        self._target_tokens = self._load_tokens(os.path.join(self._dataset_path, "target_tokens.json"))
        self._context_tokens = self._load_tokens(os.path.join(self._dataset_path, "context_tokens.json"))

        self._target_tokens += ["<start>", "<stop>", "<punkt>", "<komma>", "<space>", "<padding>"]
        self._context_tokens += ["<padding>"]

        self._stoi_targets = {token: i for i, token in enumerate(self._target_tokens)}
        self._stoi_context = {token: i for i, token in enumerate(self._context_tokens)}

        self._itos_targets = {i: token for token, i in self._stoi_targets.items()}
        self._itos_context = {i: token for token, i in self._stoi_targets.items()}

    def _load_tokens(self, filename: str) -> List:
        with open(os.path.join(self._dataset_path, filename), "r", encoding='utf-8') as f:
            return json.load(f)

    @property
    def padding_idx_context(self) -> int:
        return self._stoi_context["<padding>"]
    
    @property
    def padding_idx_target(self) -> int:
        return self._stoi_targets["<padding>"]
        
    @property
    def start_idx_target(self) -> int:
        return self._stoi_targets["<start>"]
    
    @property
    def stop_idx_target(self) -> int:
        return self._stoi_targets["<stop>"]
    
    @property
    def size_context_vocab(self) -> int:
        return max(self._stoi_context.values()) + 1
    
    @property
    def size_target_vocab(self) -> int:
        return max(self._stoi_targets.values()) + 1
    
    def stoi_targets(self, input: str) -> List[int]:
        input = input.replace(".", " <punkt>")
        input = input.replace(",", " <komma>")
        #input = input.replace(" ", " <space> ")
        
        words = input.split()

        return [self._stoi_targets[x] for x in words]
    
    def stoi_context(self, input: str) -> List[int]:        
        words = input.split(",")
        
        return [self._stoi_context[x] for x in words]

    def itos_targets(self, input: List[int]) -> str:        
        words = [self._itos_targets[x] for x in input]

        s = " ".join(words)

        #s = s.replace("<space>", " ")
        s = s.replace(" <punkt>", ".")
        s = s.replace(" <komma>", ",")

        return s
