import json
import numpy as np
import os

from enum import Enum
from random import randint
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
    def __init__(
            self, 
            path: str, 
            split: Split, 
            transformations: TransformationPipeline, 
            cached: bool = False, 
            n_samples: int = -1
        ):
        self._path = path
        self._split = split
        self._transforms = transformations
        self._cached = cached
        self._n_samples = n_samples

        # just for test purposes
        if self._n_samples == -1:
            self._full_path = os.path.join(self._path, f"dset_{self._split.value[0]}.json")
        else:
            self._full_path = os.path.join(self._path, f"dset_{self._split.value[0]}_{n_samples}.json")

        self._files = self._load_files(self._full_path)

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
    
    @staticmethod
    def _load_files(path: str) -> List[str]:
        with open(path, "r", encoding="UTF-8") as f:
            return json.load(f)
        
    def _cache_data(self) -> List[Dict]:
        cached_data = []

        for file in self._files:
            data = self._load_data_from_file(file)
            data = self._transforms(data)
            cached_data.append(data)
            
        return cached_data

    def _load_data_from_file(self, file_path: str) -> Dict:
        file_path = os.path.join(self._path, file_path)
        
        with open(file_path, 'r', encoding="UTF-8") as f:
            data = json.load(f)

        return data
    

class WeatherDatasetClassifier(Dataset):
    def __init__(
            self, 
            path: str, 
            split: Split, 
            transformations: TransformationPipeline, 
        ):
        self._path = path
        self._split = split
        self._transforms = transformations

        # just for test purposes
        self._full_path = os.path.join(self._path, f"dset_{self._split.value[0]}.json")

        self._files = self._load_files(self._full_path)

        self._cached_data = self._cache_data()

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> Dict:
        rand_int = self._draw_random_partner_idx(idx, len(self))
        
        sample = self._cached_data[idx]
        sample["class_1"] = self._cached_data[idx]["report_short_wout_boeen"]
        sample["class_0"] = self._cached_data[rand_int]["report_short_wout_boeen"]

        return sample
    
    @staticmethod
    def _load_files(path: str) -> List[str]:
        with open(path, "r", encoding="UTF-8") as f:
            return json.load(f)
        
    def _cache_data(self) -> List[Dict]:
        cached_data = []

        for file in self._files:
            data = self._load_data_from_file(file)
            data = self._transforms(data)
            cached_data.append(data)
            
        return cached_data

    def _load_data_from_file(self, file_path: str) -> Dict:
        file_path = os.path.join(self._path, file_path)
        
        with open(file_path, 'r', encoding="UTF-8") as f:
            data = json.load(f)

        return data
    
    @staticmethod
    def _draw_random_partner_idx(idx: int, n_tot: int) -> int:
        if idx == 0:
            return randint(1, n_tot-1)
        elif idx == n_tot - 1:
            return randint(0, idx - 1)
        else:
            interval_left = (0, idx)
            interval_right = (idx+1, n_tot)

            p_left = (interval_left[1] - interval_left[0]) / (n_tot - 1)
            p_right = (interval_right[1] - interval_right[0]) / (n_tot - 1)

            selected_interval = np.random.choice([0, 1], p=[p_left, p_right])

            if selected_interval == 0:
                return randint(interval_left[0], interval_left[1]-1)
            else:
                return randint(interval_right[0], interval_right[1]-1)