import unittest

import numpy as np
import torch
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.text_encoder.bpe_text_encoder import BPETextEncoder


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_bpe_encoder(self):
        text_encoder = BPETextEncoder()
        encoded_polina = text_encoder.encode("i love polina")
        print(text_encoder.ctc_decode(encoded_polina.tolist()))
        assert "i love polina" == text_encoder.ctc_decode(encoded_polina.tolist())

    def test_beam_search(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        np.random.seed(3407)
        a = np.random.random((5, 28))
        logits = np.array([a[i] / a.sum(1)[i] for i in range(len(a))])
        print(text_encoder.ctc_beam_search(torch.tensor(logits), 3))

    @staticmethod
    def build_perfect_text_probs(text, text_encoder):
        probs = torch.zeros((len(text), len(text_encoder.char2ind)))
        for i, char in enumerate(text):
            probs[i][text_encoder.char2ind[char]] = 1.
        return probs

    def test_beam_search_Polina(self):
        # test borrowed from Polina Guseva
        # true_text = "bugz bunny"
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()

        bad_probs = self.build_perfect_text_probs("bun^ny bun^ny", text_encoder)
        good_probs = self.build_perfect_text_probs("bugz^^ bun^ny", text_encoder)
        probs = 0.51 * bad_probs + 0.49 * good_probs

        decoded_beams = text_encoder.ctc_beam_search(probs, beam_size=20)
        print(decoded_beams)
        self.assertIn(decoded_beams[0][0], "bunny bunny")

    def test_russian(self):
        text_encoder = CTCCharTextEncoder.get_russian_alphabet()
        text = "я^^ ^л^ю^ббб^ллл^ю   г ^^^п^о^лллли^нууу оооо^нннн^а д^ала мммммммн^е т^^^^^^^^о^т т^е^с^тт"
        true_text = "я люблю г полину она дала мне тот тест"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)
