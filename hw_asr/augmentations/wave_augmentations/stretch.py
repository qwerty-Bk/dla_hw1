import torch

from librosa.effects import time_stretch
from hw_asr.augmentations.base import AugmentationBase


class Stretch(AugmentationBase):
    def __init__(self, r1=0.8, r2=1.2):
        self.r1 = r1
        self.r2 = r2

    def __call__(self, data: torch.Tensor):
        p = (self.r2 - self.r1) * torch.rand(1) + self.r1
        p = p.item()
        return torch.from_numpy(time_stretch(data.numpy().squeeze(), p))
