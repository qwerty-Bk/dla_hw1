from torch import nn
from hw_asr.base import BaseModel
import torch.nn.functional as F


class ResidualCNN(nn.Module):
    """Heavily inspired by https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/"""
    """Residual cnn, see https://arxiv.org/pdf/1603.05027.pdf + layer norm (taught in шад)"""

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.LayerNorm(n_feats),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(n_feats),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        )

    def forward(self, x):
        residual = x
        x = self.block1(x)
        x = self.block2(x)
        x += residual
        return x


class BidirGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class CRNNModel(BaseModel):
    def __init__(self, n_feats, n_class, n_cnn, n_rnn, hidden_size, stride=2, dropout=0.1, *args, **kwargs):
        super(CRNNModel, self).__init__(n_feats, n_class, *args, **kwargs)
        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=1)

        self.rescnn_layers = nn.Sequential()
        for i in range(n_cnn):
            self.rescnn_layers.add_module(f"rescnn {i}", ResidualCNN(32, 32, kernel=3, stride=1,
                                                                     dropout=dropout, n_feats=n_feats))

        self.fc = nn.Linear(n_feats * 32, hidden_size)
        self.birnn_layers = nn.Sequential()
        for i in range(n_rnn):
            self.birnn_layers.add_module(f"birnn {i}", BidirGRU(rnn_dim=hidden_size if i == 0 else hidden_size * 2,
                             hidden_size=hidden_size, dropout=dropout, batch_first=i == 0))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        x = spectrogram.unsqueeze(1)
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        x = x.transpose(2, 3).contiguous()
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

    def transform_input_lengths(self, input_lengths):
        return (input_lengths + 1) // 2
