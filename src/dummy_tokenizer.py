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
        with open(os.path.join(self._dataset_path, filename), "r") as f:
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

        # if word is not in _stoi_targets, add it to "target_tokens.json"
        for word in words:
            if word not in self._stoi_targets:
                # open "dataset/target_tokens.json"
                with open(os.path.join(self._dataset_path, "target_tokens.json"), "r") as f:
                    target_tokens = json.load(f)
                # add word to target_tokens
                target_tokens.append(word)
                # save target_tokens to "dataset/target_tokens.json"
                with open(os.path.join(self._dataset_path, "target_tokens.json"), "w") as f:
                    json.dump(target_tokens, f)
                # update _target_tokens
                self._target_tokens = target_tokens
                self._target_tokens += ["<start>", "<stop>", "<punkt>", "<komma>", "<space>", "<padding>"]
                # update _stoi_targets
                self._stoi_targets = {token: i for i, token in enumerate(self._target_tokens)}
                # update _itos_targets
                self._itos_targets = {i: token for token, i in self._stoi_targets.items()}

        return [self._stoi_targets[x] for x in words]
    
    def stoi_context(self, input: str) -> List[int]:        
        words = input.split(",")

        # if word is not in _stoi_context, add it to "context_tokens.json"
        for word in words:
            if word not in self._stoi_context:
                # open "dataset/context_tokens.json"
                with open(os.path.join(self._dataset_path, "context_tokens.json"), "r") as f:
                    context_tokens = json.load(f)
                # add word to context_tokens
                context_tokens.append(word)
                # save context_tokens to "dataset/context_tokens.json"
                with open(os.path.join(self._dataset_path, "context_tokens.json"), "w") as f:
                    json.dump(context_tokens, f)
                # update _context_tokens
                self._context_tokens = context_tokens
                self._context_tokens += ["<padding>"]
                # update _stoi_context
                self._stoi_context = {token: i for i, token in enumerate(self._context_tokens)}
                # update _itos_context
                self._itos_context = {i: token for token, i in self._stoi_context.items()}

        return [self._stoi_context[x] for x in words]

    def itos_targets(self, input: List[int]) -> str:        
        words = [self._itos_targets[x] for x in input]

        s = " ".join(words)

        #s = s.replace("<space>", " ")
        s = s.replace(" <punkt>", ".")
        s = s.replace(" <komma>", ",")

        return s
