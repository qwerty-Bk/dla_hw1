from typing import List, Tuple

import gdown
from torch import Tensor
import pyctcdecode
import gzip
import os, shutil, wget
import youtokentome as yttm
import numpy as np
from typing import List, Union


from hw_asr.utils import ROOT_PATH
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class BPETextEncoder(CTCCharTextEncoder):
    def __init__(self):
        data_dir = ROOT_PATH / "data" / "models"
        data_dir.mkdir(exist_ok=True, parents=True)
        model_path = data_dir / "bpe.model"
        if not os.path.exists(model_path):
            print('Downloading model for bpe.')
            gdown.download(output=str(model_path), id="1yhYOeXdltJNWCe2S4UO0ZEja6kvOsIBw")
            print('Downloaded the model.')
        bpe = yttm.BPE(model=str(model_path))
        self.model_path = model_path
        super().__init__(bpe.vocab()[1:])

    def ctc_decode(self, inds: List[int]) -> str:
        bpe = yttm.BPE(model=str(self.model_path))
        return ''.join(bpe.decode([int(i) for i in self._ctc_decode(inds)]))

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        return ''.join([self.ind2char[int(ind)] for ind in vector]).strip()

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        bpe = yttm.BPE(model=str(self.model_path))
        return Tensor(bpe.encode(text))
