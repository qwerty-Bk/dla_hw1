import gdown
import os
import zipfile
import torch
import torch.nn.functional as F
import torch_audiomentations
import torchaudio
import csv

from librosa.effects import time_stretch
from torchaudio.transforms import Vol
from hw_asr.augmentations.base import AugmentationBase
from hw_asr.utils import ROOT_PATH
from speechbrain.processing.speech_augmentation import AddNoise


class Gain(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: torch.Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)


class Stretch(AugmentationBase):
    def __init__(self, r1=0.8, r2=1.2):
        self.r1 = r1
        self.r2 = r2

    def __call__(self, data: torch.Tensor):
        return torch.from_numpy(time_stretch(data.numpy().squeeze(),
                                             (self.r2 - self.r1) * torch.rand(1) + self.r1))


class Pitch(AugmentationBase):
    def __init__(self, r1=0.8, r2=1.2):
        self.r1 = r1
        self.r2 = r2

    def __call__(self, data: torch.Tensor):
        return Vol(data, (self.r2 - self.r1) * torch.rand(1) + self.r1, gain_type='amplitude')


class Noise(AugmentationBase):
    def __init__(self):
        zip_audios = "aug_audio.zip"
        data_dir = ROOT_PATH / "data" / "augs"
        data_dir.mkdir(exist_ok=True, parents=True)
        if not os.path.exists(data_dir / zip_audios):
            print('Downloading audios for aug.')
            gdown.download(output=str(data_dir / zip_audios), id="14hLZakiTki_ncJcSwoBBtiJ__GO-B2jo")
            print('Downloaded the zipped audios.')

        audio_path = 'aug_audios'
        if not os.path.exists(data_dir / audio_path):
            print(data_dir / zip_audios)
            with zipfile.ZipFile(data_dir / zip_audios, 'r') as zip_ref:
                zip_ref.extractall(data_dir / audio_path)
            print('Unzipped the audios.')

        flac_dir = data_dir / audio_path / "aug_audio"
        paths = list(flac_dir.glob("*.wav")) + [None]
        csv_path = "aug_audio.csv"
        if not os.path.exists(data_dir / audio_path / csv_path):
            audio_csv = open(data_dir / audio_path / csv_path, 'w')
            writer = csv.writer(audio_csv)
            writer.writerows(list(map(str, paths)))
        self.csv_path = data_dir / audio_path / csv_path
        self.last_audio = 0
        self.audio_number = len(paths)

    def __call__(self, data: torch.Tensor, *args, **kwargs):
        path = self.csv_path
        if self.last_audio + 1 == self.audio_number: # white noise
            path = None
        noisifier = AddNoise(path, csv_keys=self.last_audio,
                             sorting='original', normalize=True, pad_noise=True)
        self.last_audio = (self.last_audio + 1) % self.audio_number
        return noisifier(data, torch.ones(1))


class Impulse(AugmentationBase):
    def __init__(self):
        room_audio = "room.wav"
        data_dir = ROOT_PATH / "data" / "augs"
        data_dir.mkdir(exist_ok=True, parents=True)
        if not os.path.exists(data_dir / room_audio):
            print('Downloading audios for aug.')
            gdown.download(output=str(data_dir / room_audio), id="1es0v3-SKYMxW_dbQjab1aexz2F-qPldX")
            print('Downloaded the room audios.')

        self.path = data_dir / room_audio

    def __call__(self, data: torch.Tensor, *args, **kwargs):
        room, _ = torchaudio.load(self.path)
        left_pad = right_pad = room.shape[-1] - 1
        flipped_rir = room.squeeze().flip(0)
        audio = F.pad(data, [left_pad, right_pad]).view(1, 1, -1)
        convolved_audio = torch.conv1d(audio, flipped_rir.view(1, 1, -1)).squeeze()
        if convolved_audio.abs().max() > 1:
            convolved_audio /= convolved_audio.abs().max()
        return convolved_audio
