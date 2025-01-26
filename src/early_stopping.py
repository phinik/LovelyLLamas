import json
import os

from typing import Any, Optional
from enum import Enum


class OptDirection(Enum):
    MAXIMIZE = 0
    MINIMIZE = 1


# TODO: write tests
class EarlyStopping:
    def __init__(self, direction: OptDirection, patience: int = 10, save_path: Optional[str] = None):
        self._opt_direction: OptDirection = direction
        self._patience: int = patience
        self._save_path: Optional[str] = save_path

        self._best_value: Optional[Any] = None
        self._best_value_epoch: Optional[int] = None
        
    def update(self, metric_value: Any, epoch: int) -> bool:
        if self._current_model_is_better(metric_value):            
            self._reset_patience(metric_value, epoch)            
            return False

        elif self._patience_is_triggered(epoch):
            self._save_early_stopping_note(epoch)           
            return True
                
        return False  # Case: current model is not better, but patience has not been triggered yet

    def _current_model_is_better(self, metric_value: Any) -> bool:
        if self._best_value is None or self._best_value_epoch is None:
            return True
        
        if self._opt_direction == OptDirection.MAXIMIZE and metric_value > self._best_value:
            return True
        
        if self._opt_direction == OptDirection.MINIMIZE and metric_value < self._best_value:
            return True
            
        return False
    
    def _reset_patience(self, metric_value: Any, epoch: int) -> None:
        self._best_value = metric_value
        self._best_value_epoch = epoch
    
    def _patience_is_triggered(self, epoch) -> bool:
        return (epoch - self._best_value_epoch) >= self._patience
    
    def _save_early_stopping_note(self, epoch: int) -> None:
        if self._save_path is not None:
            with open(os.path.join(self._save_path, "early_stopping.json"), 'w') as f:
                json.dump({"epoch": epoch}, f)
