from typing import Any, Optional
from enum import Enum
import json
import os


class OptDirection(Enum):
    MAXIMIZE = 0
    MINIMIZE = 1


# TODO: write tests
class BestModel:
    def __init__(self, metric_name: str, direction: OptDirection, save_path: str):
        self._best_value: Optional[Any] = None
        self._metric_name: str = metric_name
        self._opt_direction: OptDirection = direction
        self._save_path: str = save_path

        os.makedirs(self._save_path, exist_ok=True)

    def update(self, epoch: int, model, metric_value: Any) -> None:
        if self._current_model_is_better(metric_value):
            model.save_weights_as(self._save_path, f"best_model_{self._metric_name}")

            metadata = {"epoch": epoch, self._metric_name: metric_value}
            with open(os.path.join(self._save_path, f"best_model_{self._metric_name}_metadata.json"), "w") as f:
                json.dump(metadata, f, sort_keys=True, indent=4)
            
            self._best_value = metric_value

    def _current_model_is_better(self, metric_value: Any) -> bool:
        if self._best_value is None:
            return True
        
        if self._opt_direction == OptDirection.MAXIMIZE and metric_value > self._best_value:
            return True
        
        if self._opt_direction == OptDirection.MINIMIZE and metric_value < self._best_value:
            return True
            
        return False