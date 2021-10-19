from typing import List, Callable

from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class SequentialAugmentation(AugmentationBase):
    def __init__(self, augmentation_list: List[Callable]):
        self.augmentation_list = []
        for aug in augmentation_list:
            self.augmentation_list.append(aug)

    def __call__(self, data: Tensor) -> Tensor:
        x = data
        for augmentation in self.augmentation_list:
            x = augmentation(x)
        return x
