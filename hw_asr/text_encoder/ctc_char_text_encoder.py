from typing import List, Tuple

import torch
import pyctcdecode
import gzip
import os, shutil, wget

from hw_asr.utils import ROOT_PATH
from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        lm_path, unigram_list = self.prepare_kenlm()
        self.bs_decoder = pyctcdecode.decoder.build_ctcdecoder(
            [''] + alphabet, lm_path, unigram_list
        )

    def ctc_decode(self, inds: List[int]) -> str:
        res = ""
        prev_empty = False
        for ind in inds:
            if ind == 0:
                prev_empty = True
            else:
                if len(res) == 0 or res[-1] != self.ind2char[ind] or res[-1] == self.ind2char[ind] and prev_empty:
                    res += self.ind2char[ind]
                prev_empty = False
        return res

    def prepare_kenlm(self):
        """inspired by the following tutorial to kenlm:
        https://github.com/kensho-technologies/pyctcdecode/blob/main/tutorials/01_pipeline_nemo.ipynb"""

        gz_three_gram_path = "3-gram.pruned.1e-7.arpa.gz"
        data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
        data_dir.mkdir(exist_ok=True, parents=True)
        if not os.path.exists(data_dir / gz_three_gram_path):
            print('Downloading pruned 3-gram model.')
            lm_url = 'http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz'
            lm_gzip_path = wget.download(lm_url, out=str(data_dir))
            print('Downloaded the 3-gram language model.')
        else:
            print('Pruned .arpa.gz already exists.')

        upper_lm_path = '3-gram.pruned.1e-7.arpa'
        if not os.path.exists(data_dir / upper_lm_path):
            with gzip.open(lm_gzip_path, 'rb') as f_zipped:
                with open(data_dir / upper_lm_path, 'wb') as f_unzipped:
                    shutil.copyfileobj(f_zipped, f_unzipped)
            print('Unzipped the 3-gram language model.')
        else:
            print('Unzipped .arpa already exists.')

        lm_path = 'lowercase_3-gram.pruned.1e-7.arpa'
        if not os.path.exists(data_dir / lm_path):
            with open(data_dir / upper_lm_path, 'r') as f_upper:
                with open(data_dir / lm_path, 'w') as f_lower:
                    for line in f_upper:
                        f_lower.write(line.lower().replace("'", ""))
        print('Converted language model file to lowercase.')

        lib_vocab = "librispeech-vocab.txt"
        if not os.path.exists(data_dir / lib_vocab):
            ls_url = "http://www.openslr.org/resources/11/librispeech-vocab.txt"
            wget.download(ls_url, out=str(data_dir))
            print('Downloaded librispeech vocabulary')

        with open(data_dir / lib_vocab) as f:
            unigram_list = [t.lower().replace("'", "") for t in f.read().strip().split("\n")]

        return str(data_dir / lm_path), unigram_list

    def ctc_beam_search(self, probs: torch.tensor, beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        # TODO: your code here
        hypos = self.bs_decoder.decode_beams(probs.cpu().detach().numpy(),
                                             beam_width=beam_size)  # , token_min_logp=-np.inf)

        return [(x[0], x[4]) for x in hypos]
