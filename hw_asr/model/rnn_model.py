from torch import nn, zeros
from hw_asr.base import BaseModel

import torch


class RnnModel(BaseModel):
    def __init__(self, n_feats, n_class, layer_dim, fc_hidden=512, *args, **kwargs):
        super(RnnModel, self).__init__(n_feats, n_class, *args, **kwargs)

        self.fc_hidden = fc_hidden
        self.layer_dim = layer_dim

        self.rnn = nn.RNN(n_feats, fc_hidden, layer_dim, batch_first=True)
        self.fc = nn.Linear(fc_hidden, n_class)

    def forward(self, spectrogram, *args, **kwargs):
        h0 = zeros(self.layer_dim, spectrogram.shape[0], self.fc_hidden).requires_grad_()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output, h0 = self.rnn(spectrogram, h0.to(device))
        return self.fc(output) # [:, -1, :])

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
