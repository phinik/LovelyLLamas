import json
import os

from abc import ABC, abstractmethod
from typing import List


from .itokenizer import ITokenizer


class SetOfWordsTokenizer(ITokenizer):
    _SPECIAL_TOKENS = ["<city>", "<start>", "<stop>", "<unknown>", "<padding>"]

    def __init__(self, tokens_file: str, extra_tokens: List[str]):
        super().__init__()

        self._tokens = self._load_tokens(tokens_file)

        self._tokens += self._SPECIAL_TOKENS
        self._tokens += extra_tokens
        self._tokens = sorted(list(set(self._tokens)))
        
        self._stoi = {token: i for i, token in enumerate(self._tokens)}
        self._itos = {i: token for token, i in self._stoi.items()}

    def _load_tokens(self, tokens_file: str) -> List:
        with open(tokens_file, "r") as f:
            return json.load(f)

    @abstractmethod
    def _insert_extra_tokens(self, s: str) -> str:
        ...

    @abstractmethod
    def _remove_extra_tokens(self, s: str) -> str:
        ...

    def add_start_stop_tokens(self, s: str) -> str:
        return f"<start> {s} <stop>"
        
    @property
    def padding_idx(self) -> int:
        return self._stoi["<padding>"]
        
    @property
    def start_idx(self) -> int:
        return self._stoi["<start>"]
    
    @property
    def stop_idx(self) -> int:
        return self._stoi["<stop>"]
    
    @property
    def unknown_idx(self) -> int:
        return self._stoi["<unknown>"]
        
    @property
    def vocab_size(self) -> int:
        return len(self._stoi)
        
    def stoi(self, input: str) -> List[int]:
        s = self._insert_extra_tokens(input)
        
        words = s.split()

        return [self._stoi.get(x, self._stoi["<unknown>"]) for x in words]
    
    def itos(self, input: List[int]) -> str:        
        words = [self._itos[x] for x in input]

        s = " ".join(words)
        s = self._remove_extra_tokens(s)

        return s
    

class SetOfWordsTokenizerRepShort(SetOfWordsTokenizer):
    _MAPPING = {
            ".": "<point>",
            ",": "<comma>"
        }
    
    def __init__(self, dataset_path: str):
        super().__init__(
            tokens_file=os.path.join(dataset_path, "target_tokens.json"),
            extra_tokens=list(self._MAPPING.values())
        )

    def _insert_extra_tokens(self, s: str) -> str:
        for k, v in self._MAPPING.items():
            s = s.replace(k, f" {v}")
    
        return s

    def _remove_extra_tokens(self, s: str) -> str:
        for k, v in self._MAPPING.items():
            s = s.replace(f" {v}", k)
            
        return s

class SetOfWordsTokenizerGPT(SetOfWordsTokenizer):
    _MAPPING = {
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
    
    def __init__(self, dataset_path: str):
        super().__init__(
            tokens_file=os.path.join(dataset_path, "target_tokens_gpt_filtered.json"), 
            extra_tokens=list(self._MAPPING.values())
        )

    def _insert_extra_tokens(self, s: str) -> str:
        for k, v in self._MAPPING.items():
            s = s.replace(k, f" {v}")
    
        return s

    def _remove_extra_tokens(self, s: str) -> str:
        for k, v in self._MAPPING.items():
            s = s.replace(f" {v}", k)

        return s
