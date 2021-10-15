import unittest

import torch

from hw_asr.collate_fn.collate import collate_fn
from hw_asr.datasets import LibrispeechDataset
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.utils.parse_config import ConfigParser


class TestDataloader(unittest.TestCase):
    def test_collate_fn(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        config_parser = ConfigParser.get_default_configs()

        ds = LibrispeechDataset(
            "dev-clean", text_encoder=text_encoder, config_parser=config_parser
        )

        BS = 3
        batch = collate_fn([ds[i] for i in range(BS)])

        self.assertIn("spectrogram", batch)  # torch.tensor
        bs, audio_time_length, feature_length = batch["spectrogram"].shape
        self.assertEqual(bs, BS)
        # print('audio length %d, feature length %d'%(audio_time_length, feature_length))

        self.assertIn("text_encoded", batch)  # [int] torch.tensor
        # joined and padded indexes representation of transcriptions
        bs, text_time_length = batch["text_encoded"].shape
        self.assertEqual(bs, BS)

        self.assertIn("text_encoded_length", batch)  # [int] torch.tensor
        # contains lengths of each text entry
        self.assertEqual(len(batch["text_encoded_length"].shape), 1)
        bs = batch["text_encoded_length"].shape[0]
        self.assertEqual(bs, BS)

        self.assertIn("text", batch)  # List[str]
        # simple list of initial normalized texts
        bs = len(batch["text"])
        self.assertEqual(bs, BS)

        return batch

    def test_collate_fn_small(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        config_parser = ConfigParser.get_default_configs()

        BS = 3
        a = {
            'audio': torch.Tensor([[1., 2.]]),
            'spectrogram': torch.Tensor([[[1.], [2.]]]),
            'duration': 1.,
            'text': 'a',
            'text_encoded': torch.Tensor([[1]]),
            'audio_path': 'c'
        }
        b = {
            'audio': torch.Tensor([[3., 4., 5., 6.]]),
            'spectrogram': torch.Tensor([[[3., 4.], [5., 6.]]]),
            'duration': 2.,
            'text': 'b',
            'text_encoded': torch.Tensor([[2, 3]]),
            'audio_path': 'b'
        }
        c = {
            'audio': torch.Tensor([[7., 8., 9.]]),
            'spectrogram': torch.Tensor([[[7.], [8.]]]),
            'duration': 3.,
            'text': 'c',
            'text_encoded': torch.Tensor([[4, 5, 6]]),
            'audio_path': 'a'
        }

        batch = collate_fn([a, b, c])
        print(batch)

        self.assertIn("spectrogram", batch)  # torch.tensor
        bs, audio_time_length, feature_length = batch["spectrogram"].shape
        self.assertEqual(bs, BS)

        self.assertIn("text_encoded", batch)  # [int] torch.tensor
        # joined and padded indexes representation of transcriptions
        bs, text_time_length = batch["text_encoded"].shape
        self.assertEqual(bs, BS)

        self.assertIn("text_encoded_length", batch)  # [int] torch.tensor
        # contains lengths of each text entry
        self.assertEqual(len(batch["text_encoded_length"].shape), 1)
        bs = batch["text_encoded_length"].shape[0]
        self.assertEqual(bs, BS)

        self.assertIn("text", batch)  # List[str]
        # simple list of initial normalized texts
        bs = len(batch["text"])
        self.assertEqual(bs, BS)

        return batch
