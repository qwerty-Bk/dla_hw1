import torchaudio.transforms
import torch

from hw_asr.augmentations.base import AugmentationBase


class SpecAug(AugmentationBase):
    def __init__(self):
        self.specaug = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(20),
            torchaudio.transforms.TimeMasking(100),
        )

    def __call__(self, log_mel: torch.Tensor, *args, **kwargs):
        return self.specaug(log_mel)
