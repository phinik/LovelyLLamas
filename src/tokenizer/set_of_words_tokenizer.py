import json
import os

from abc import ABC, abstractmethod
from typing import List


from .itokenizer import ITokenizer


class SetOfWordsTokenizer(ITokenizer):
    SPECIAL_TOKENS = ["<city>", "<start>", "<stop>", "<unknown>", "<padding>"]

    def __init__(self, dataset_path: str):
        super().__init__()

        self._dataset_path = dataset_path

        self._target_tokens = self._load_tokens(os.path.join(self._dataset_path, self._target_tokens_file))
        self._context_tokens = self._load_tokens(os.path.join(self._dataset_path, "context_tokens.json"))

        self._target_tokens += self.SPECIAL_TOKENS
        self._target_tokens += self._extra_tokens

        self._target_tokens = sorted(list(set(self._target_tokens).union(set(self._context_tokens))))
        #self._context_tokens += ["<padding>"]
        self._context_tokens = self._target_tokens

        self._stoi_targets = {token: i for i, token in enumerate(self._target_tokens)}
        self._stoi_context = {token: i for i, token in enumerate(self._context_tokens)}

        self._itos_targets = {i: token for token, i in self._stoi_targets.items()}

    def _load_tokens(self, filename: str) -> List:
        with open(os.path.join(self._dataset_path, filename), "r") as f:
            return json.load(f)

    @property
    @abstractmethod
    def _target_tokens_file(self) -> str:
        ...

    @property
    @abstractmethod
    def _extra_tokens(self) -> List[str]:
        ...

    @abstractmethod
    def _insert_extra_tokens(self, s: str) -> str:
        ...

    @abstractmethod
    def _remove_extra_tokens(self, s: str) -> str:
        ...

    def add_start_stop_tokens(self, s: str) -> str:
        return f"<start> {s} <stop>"
    
    @property
    def padding_idx_context(self) -> int:
        return self._stoi_context["<padding>"]
    
    @property
    def padding_idx(self) -> int:
        return self._stoi_targets["<padding>"]
        
    @property
    def start_idx(self) -> int:
        return self._stoi_targets["<start>"]
    
    @property
    def stop_idx(self) -> int:
        return self._stoi_targets["<stop>"]
    
    @property
    def unknown_idx(self) -> int:
        return self._stoi_targets["<unknown>"]
    
    @property
    def size_context_vocab(self) -> int:
        return max(self._stoi_context.values()) + 1
    
    @property
    def vocab_size(self) -> int:
        return max(self._stoi_context.values()) + 1
        
    def stoi(self, input: str) -> List[int]:
        s = self._insert_extra_tokens(input)
        
        words = s.split()

        return [self._stoi_targets.get(x, self._stoi_targets["<unknown>"]) for x in words]
    
    def stoi_context(self, input: str) -> List[int]:        
        words = input.split(";")

        return [self._stoi_context.get(x, self._stoi_context["<unknown>"]) for x in words]

    def itos(self, input: List[int]) -> str:        
        words = [self._itos_targets[x] for x in input]

        s = " ".join(words)

        s = self._remove_extra_tokens(s)

        return s
    

class SetOfWordsTokenizerRepShort(SetOfWordsTokenizer):
    def __init__(self, dataset_path: str):
        self._mapping = {
            ".": "<point>",
            ",": "<comma>"
        }

        super().__init__(dataset_path)

    @property
    def _target_tokens_file(self) -> str:
        return "target_tokens.json"
        
    @property
    def _extra_tokens(self) -> List[str]:
        return list(self._mapping.values())

    def _insert_extra_tokens(self, s: str) -> str:
        for k, v in self._mapping.items():
            s = s.replace(k, f" {v}")
    
        return s

    def _remove_extra_tokens(self, s: str) -> str:
        for k, v in self._mapping.items():
            s = s.replace(f" {v}", k)
            
        return s

class SetOfWordsTokenizerGPT(SetOfWordsTokenizer):
    def __init__(self, dataset_path: str):
        self._mapping = {
            ".": "<point>",
            ",": "<comma>",
            "!": "<exclamationMark>",
            "\"": "<quote>",
            ":": " <colon>",
            "?": " <questionMark>",
            "(": " <openBraket>",
            ")": " <closeBraket>",
            ";": " <semicolon>",
        }

        super().__init__(dataset_path)

    @property
    def _target_tokens_file(self) -> str:
        return "target_tokens_gpt_filtered.json"

    @property
    def _extra_tokens(self) -> List[str]:
        return list(self._mapping.values())

    def _insert_extra_tokens(self, s: str) -> str:
        for k, v in self._mapping.items():
            s = s.replace(k, f" {v}")
    
        return s

    def _remove_extra_tokens(self, s: str) -> str:
        for k, v in self._mapping.items():
            s = s.replace(f" {v}", k)

        return s
