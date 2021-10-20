from hw_asr.utils import ROOT_PATH
import youtokentome as yttm
from hw_asr.base.base_text_encoder import BaseTextEncoder


if __name__ == "__main__":
    libri_path = ROOT_PATH / "data" / "datasets" / "librispeech" / "train-clean-100"
    paths = list(libri_path.rglob("*.txt"))
    corpus_path = "data/train_corpus.txt"
    train_txt = open(corpus_path, 'w')
    for path in paths:
        file = open(path, 'r')
        for line in file:
            train_txt.write(BaseTextEncoder.normalize_text(line).replace("'", ''))
            train_txt.write('\n')
    vs = 50
    yttm.BPE.train(data=corpus_path, vocab_size=vs, model="data/bpe.model")
