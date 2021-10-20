import json
from pathlib import Path
from string import ascii_lowercase
from typing import List, Union

import numpy as np
from torch import Tensor
from bpemb import BPEmb

from hw_asr.base.base_text_encoder import BaseTextEncoder


class CharTextEncoder(BaseTextEncoder):

    def __init__(self, alphabet: List[str], bpe: bool, bpe_size: int = 0):
        self.ind2char = {k: v for k, v in enumerate(sorted(alphabet))}
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.bpe = bpe
        if bpe:
            self.bpe_encoder = BPEmb("en", vs=bpe_size)

    def __len__(self):
        return len(self.ind2char)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        try:
            if not self.bpe:
                return Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
            else:
                return Tensor([self.char2ind[char] for char in self.bpe_encoder.encode(text)]).unsqueeze(0)
        except KeyError as e:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'")

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        return ''.join([self.ind2char[int(ind)] for ind in vector]).strip()

    def dump(self, file):
        with Path(file).open('w') as f:
            json.dump(self.ind2char, f)

    @classmethod
    def from_file(cls, file):
        with Path(file).open() as f:
            ind2char = json.load(f)
        a = cls([], True, len(ind2char))
        a.ind2char = ind2char
        a.char2ind = {v: k for k, v in ind2char}
        return a

    @classmethod
    def get_simple_alphabet(cls):
        return cls(alphabet=list(ascii_lowercase + ' '), bpe=False)
