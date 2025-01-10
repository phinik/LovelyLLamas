import json
import os


from typing import List, Dict


from .itokenizer import ITokenizer


class SetOfWordsTokenizer(ITokenizer):
    _SPECIAL_TOKENS = ["<city>", "<start>", "<stop>", "<unknown>", "<padding>"]

    def __init__(self, tokens_file: str, extra_tokens_mapping: Dict):
        super().__init__()

        self._tokens = self._load_tokens(tokens_file)
        self._tokens += self._SPECIAL_TOKENS
        self._tokens += list(extra_tokens_mapping.values())
        self._tokens = sorted(list(set(self._tokens)))
        
        self._stoi = {token: i for i, token in enumerate(self._tokens)}
        self._itos = {i: token for token, i in self._stoi.items()}

        self._extra_tokens_mapping = extra_tokens_mapping

    def _load_tokens(self, tokens_file: str) -> List:
        with open(tokens_file, "r") as f:
            return json.load(f)

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
    
    def _insert_extra_tokens(self, s: str) -> str:
        for k, v in self._MAPPING.items():
            s = s.replace(k, f" {v}")
    
        return s
    
    def itos(self, input: List[int]) -> str:        
        words = [self._itos[x] for x in input]

        s = " ".join(words)
        s = self._remove_extra_tokens(s)

        return s
    
    def _remove_extra_tokens(self, s: str) -> str:
        for k, v in self._MAPPING.items():
            s = s.replace(f" {v}", k)
            
        return s
    

class SetOfWordsTokenizerRepShort(SetOfWordsTokenizer):
    _MAPPING = {
            ".": "<point>",
            ",": "<comma>"
        }
    
    def __init__(self, dataset_path: str):
        super().__init__(
            tokens_file=os.path.join(dataset_path, "target_tokens.json"),
            extra_tokens_mapping=self._MAPPING
        )
    

class SetOfWordsTokenizerGPT(SetOfWordsTokenizer):
    _MAPPING = {
            ".": "<point>",
            ",": "<comma>",
            "!": "<exclamationMark>",
            "\"": "<quote>",
            ":": "<colon>",
            "?": "<questionMark>",
            "(": "<openBraket>",
            ")": "<closeBraket>",
            ";": "<semicolon>",
        }
    
    def __init__(self, dataset_path: str):
        super().__init__(
            tokens_file=os.path.join(dataset_path, "target_tokens_gpt_filtered.json"), 
            extra_tokens_mapping=self._MAPPING
        )
