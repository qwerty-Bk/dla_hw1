import torch

from torchaudio.transforms import Vol
from hw_asr.augmentations.base import AugmentationBase


class Pitch(AugmentationBase):
    def __init__(self, r1=0.3, r2=3):
        self.r1 = r1
        self.r2 = r2

    def __call__(self, data: torch.Tensor):
        p = (self.r2 - self.r1) * torch.rand(1) + self.r1
        p = p.item()
        return Vol(p)(data)
