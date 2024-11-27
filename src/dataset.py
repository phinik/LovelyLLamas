import json
import os
from enum import Enum
from typing import Dict, List
from torch.utils.data import Dataset


class TransformationPipeline:
    """
    Pipeline to sequentially apply a series of transformations to data.
    """
    def __init__(self, transforms: List):
        self._transforms = transforms

    def __call__(self, data: Dict) -> Dict:
        for transform in self._transforms:
            data = transform(data)
        return data
    

class Split(Enum):
    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"


class WeatherDataset(Dataset):
    """
    Custom Dataset for handling weather data.
    """
    def __init__(self, path: str, split: Split, transformations: TransformationPipeline, cached: bool = False):
        self.path = path
        self.split = split
        self.transforms = transformations
        self.cached = cached

        self.full_path = os.path.join(self.path, self.split.value)
        if not os.path.exists(self.full_path):
            raise FileNotFoundError(f"Dataset path not found: {self.full_path}")

        self.files = [f for f in os.listdir(self.full_path) if f.endswith('.json')]
        if not self.files:
            raise ValueError(f"No JSON files found in the directory: {self.full_path}")

        self.cached_data = self._cache_data() if self.cached else None

    def __len__(self) -> int:
        return len(self.cached_data) if self.cached else len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        if self.cached_data is not None:
            return self.cached_data[idx]
        return self._process_file(self.files[idx])

    def _cache_data(self) -> List[Dict]:
        return [self._process_file(file) for file in self.files]
    
    def _process_file(self, filename: str) -> Dict:
        data = self._load_data_from_file(filename)
        return self.transforms(data)

    def _load_data_from_file(self, filename: str) -> Dict:
        file_path = os.path.join(self.full_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON file {filename}: {e}")
