import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_asr.base.base_dataset import BaseDataset
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)

URL_LINKS = {
    "train-rus": "https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/public_youtube1120_hq.tar.gz",
    "test-rus": "https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/public_youtube700_val.tar.gz"
}


class RussianDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        assert part in URL_LINKS

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "russian"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        if not os.path.exists(arch_path):
            download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "Russian").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "Russian"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index_ru.json"
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
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        1 / 0
        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".flac") for f in filenames]):
                flac_dirs.add(dirpath)
        for flac_dir in tqdm(
                list(flac_dirs), desc=f"Preparing russian lang folders: {part}"
        ):
            flac_dir = Path(flac_dir)
            trans_path = list(flac_dir.glob("*.trans.txt"))[0]
            with trans_path.open() as f:
                for line in f:
                    f_id = line.split()[0]
                    f_text = " ".join(line.split()[1:]).strip()
                    flac_path = flac_dir / f"{f_id}.flac"
                    t_info = torchaudio.info(str(flac_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(flac_path.absolute().resolve()),
                            "text": f_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index


if __name__ == "__main__":
    text_encoder = CTCCharTextEncoder.get_russian_alphabet()
    config_parser = ConfigParser.get_default_configs()

    ds = RussianDataset(
        "train-rus", text_encoder=text_encoder, config_parser=config_parser
    )
    item = ds[0]
    print(item)

    ds = RussianDataset(
        "test-rus", text_encoder=text_encoder, config_parser=config_parser
    )
    item = ds[0]
    print(item)
