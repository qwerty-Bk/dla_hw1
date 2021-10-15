from torch import nn, zeros

from hw_asr.base import BaseModel


class LstmModel(BaseModel):
    def __init__(self, n_feats, n_class, layer_dim, fc_hidden=512, *args, **kwargs):
        super(LstmModel, self).__init__(n_feats, n_class, *args, **kwargs)

        self.layer_dim = layer_dim
        self.fc_hidden = fc_hidden

        self.lstm = nn.LSTM(n_feats, fc_hidden, layer_dim)
        self.fc = nn.Linear(fc_hidden, n_class)

    def forward(self, spectrogram, *args, **kwargs):
        # print(spectrogram.shape)
        h0 = zeros(self.layer_dim, spectrogram.shape[1], self.fc_hidden).requires_grad_()
        c0 = zeros(self.layer_dim, spectrogram.shape[1], self.fc_hidden).requires_grad_()

        output, _ = self.lstm(spectrogram, (h0.detach(), c0.detach()))
        res = self.fc(output)  # [:, -1, :])

        # print(res.shape)
        return res

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
