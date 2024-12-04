import torch.nn as nn


class CELoss(nn.Module):
    def __init__(self, ignore_idx: int):
        super().__init__()

        self._loss = nn.CrossEntropyLoss(ignore_index=ignore_idx, reduction="sum")

    def forward(self, prediction, target):
        return self._loss(prediction, target)