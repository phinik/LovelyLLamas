import json
import os

from enum import Enum
from typing import Dict

from torch.utils.data import Dataset

class Split(Enum):
    TRAIN = "train",
    EVAL = "eval",
    TEST = "test"


class WeatherDataset(Dataset):
    def __init__(self, path: str, split: Split):
        self._path = path
        self._split = split

        # just for test purposes
        self._full_path = os.path.join(self._path, self._split.value[0])

        self._files = os.listdir(self._full_path)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx) -> Dict:
        file_path = os.path.join(self._full_path, self._files[idx])
        
        with open(file_path, 'r') as f:
            data = json.load(f)

        return data

