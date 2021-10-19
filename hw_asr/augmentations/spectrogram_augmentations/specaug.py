import torchaudio.transforms
import torch

from hw_asr.augmentations.base import AugmentationBase


class SpecAug(AugmentationBase):
    def __init__(self, *args, **kwargs):
        freq_mask, time_mask = 100, 20
        if "freq_mask" in kwargs.keys():
            freq_mask = kwargs["freq_mask"]
        if "time_mask" in kwargs.keys():
            time_mask = kwargs["time_mask"]
        self.specaug = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask),
            torchaudio.transforms.TimeMasking(time_mask),
        )

    def __call__(self, log_mel: torch.Tensor, *args, **kwargs):
        return self.specaug(log_mel).squeeze()
