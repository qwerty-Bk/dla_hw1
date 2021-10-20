import json
import logging
import os
import shutil
from pathlib import Path

import librosa
import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_asr.base.base_dataset import BaseDataset
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)

URL_LINK = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


class LJDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self.name = "LJSpeech-1.1"

        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_part(self):
        arch_path = self._data_dir / Path(self.name + ".tar.bz2")
        print("Loading LJ Dataset")
        if not os.path.exists(arch_path):
            download_file(URL_LINK, arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / self.name).iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / self.name))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index_lj.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / "metadata.csv"
        if not split_dir.exists():
            self._load_part()

        data_dir = self._data_dir / "wavs"
        with open(split_dir, 'r', encoding="utf8") as f:
            for i, line in enumerate(f):
                if not (part == 'test' and i % 10 != 0 or part == 'train' and i % 10 != 0):
                    file_id, text, _ = line.split('|')
                    wav = data_dir / Path(file_id + ".wav")
                    length = librosa.get_duration(filename=wav)
                    index.append(
                        {
                            "path": str(wav),
                            "text": BaseTextEncoder.normalize_text(text),
                            "audio_len": length
                        }
                    )
        return index


if __name__ == "__main__":
    text_encoder = CTCCharTextEncoder.get_simple_alphabet()
    config_parser = ConfigParser.get_default_configs()

    ds = LJDataset(
        "dev-clean", text_encoder=text_encoder, config_parser=config_parser
    )
    item = ds[0]
    print(item)
