import json
import os

from enum import Enum
from typing import Dict, List

from torch.utils.data import Dataset


class TransformationPipeline:
    def __init__(self, transforms: List):
        self._transforms = transforms

    def __call__(self, data: Dict) -> Dict:
        for transform in self._transforms:
            data = transform(data)

        return data
    

class Split(Enum):
    TRAIN = "train",
    EVAL = "eval",
    TEST = "test",


class WeatherDataset(Dataset):
    def __init__(self, path: str, split: Split, transformations: TransformationPipeline, cached: bool = False):
        self._path = path
        self._split = split
        self._transforms = transformations
        self._cached = cached

        # just for test purposes
        self._full_path = os.path.join(self._path, self._split.value[0])

        self._files = os.listdir(self._full_path)

        self._cached_data = None
        if self._cached:
            self._cached_data = self._cache_data()

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> Dict:
        if self._cached_data:
            return self._cached_data[idx]
        else:
            data = self._load_data_from_file(self._files[idx])
            data = self._transforms(data)

            return data
    
    def _cache_data(self) -> List[Dict]:
        cached_data = []

        for file in self._files:
            data = self._load_data_from_file(file)
            data = self._transforms(data)
            cached_data.append(data)

        return cached_data

    def _load_data_from_file(self, filename: str) -> Dict:
        file_path = os.path.join(self._full_path, filename)
        
        with open(file_path, 'r') as f:
            data = json.load(f)

        return data
    
    def _apply_transforms(self, data: Dict) -> Dict:
        for transform in self._transforms:
            data = transform(data)

        return data


