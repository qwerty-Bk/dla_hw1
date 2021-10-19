import gdown
import os
import zipfile
import torch
import librosa

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.utils import ROOT_PATH


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
        self.paths = list(flac_dir.glob("*.wav"))
        self.last_audio = 0
        self.audio_number = len(self.paths)

    def __call__(self, wav: torch.Tensor, *args, **kwargs):
        wav = wav.squeeze()
        noise, _ = librosa.load(self.paths[self.last_audio])
        self.last_audio = (self.last_audio + 1) % self.audio_number
        noize_level = torch.randint(20, size=(1,))

        noize_energy = torch.norm(torch.from_numpy(noise))
        audio_energy = torch.norm(wav)

        alpha = (audio_energy / noize_energy) * torch.pow(10, -noize_level / 20)

        if wav.shape[0] > noise.shape[0]:
            clipped_wav = wav[..., :noise.shape[0]]
            clipped_noise = noise
        else:
            clipped_wav = wav
            clipped_noise = noise[..., :wav.shape[0]]

        augumented_wav = clipped_wav + alpha * torch.from_numpy(clipped_noise)

        if wav.shape[0] > noise.shape[0]:
            augumented_wav = torch.cat((augumented_wav, wav[..., noise.shape[0]:]))
        augumented_wav = torch.clamp(augumented_wav, -1, 1)

        return augumented_wav.unsqueeze(0)



