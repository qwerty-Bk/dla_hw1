import torch
import os
import gdown
import torchaudio
import torch.nn.functional as F

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.utils import ROOT_PATH


class Impulse(AugmentationBase):
    def __init__(self):
        room_audio = "room.wav"
        data_dir = ROOT_PATH / "data" / "augs"
        data_dir.mkdir(exist_ok=True, parents=True)
        if not os.path.exists(data_dir / room_audio):
            print('Downloading audios for aug.')
            gdown.download(output=str(data_dir / room_audio), id="1es0v3-SKYMxW_dbQjab1aexz2F-qPldX")
            print('Downloaded the room audio.')

        self.room = torchaudio.load(data_dir / room_audio)

    def __call__(self, data: torch.Tensor, *args, **kwargs):
        left_pad = right_pad = self.room.shape[-1] - 1
        flipped_rir = self.room.squeeze().flip(0)
        audio = F.pad(data, [left_pad, right_pad]).view(1, 1, -1)
        convolved_audio = torch.conv1d(audio, flipped_rir.view(1, 1, -1)).squeeze()
        if convolved_audio.abs().max() > 1:
            convolved_audio /= convolved_audio.abs().max()
        return convolved_audio
