import torch

from hw_asr.augmentations.base import AugmentationBase
import random


class SpecAug(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.masks = [kwargs.get("freq_mask", 20), kwargs.get("time_mask", 100)]
        self.repeat = kwargs.get("repeat", 2)

    def _mask(self, spec, axis):
        for i in range(self.repeat):
            mask = random.randrange(0, self.masks[axis])
            mask_zero = random.randrange(0, spec.shape[1 + axis] - mask)
            if mask_zero == mask_zero + mask:
                return spec
            mask_end = random.randrange(mask_zero, mask_zero + mask)
            spec[0][mask_zero:mask_end] = spec.mean()
        return spec

    def __call__(self, log_mel: torch.Tensor, *args, **kwargs):
        res = self._mask(log_mel, 0)
        return self._mask(res, 0)
