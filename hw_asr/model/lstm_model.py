from torch import nn
from hw_asr.base import BaseModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LstmModel(BaseModel):
    def __init__(self, n_feats, n_class, layer_dim, fc_hidden=512, *args, **kwargs):
        super(LstmModel, self).__init__(n_feats, n_class, *args, **kwargs)

        self.layer_dim = layer_dim
        self.fc_hidden = fc_hidden

        self.lstm = nn.LSTM(n_feats, fc_hidden, layer_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(fc_hidden * 2, n_class)

    def forward(self, spectrogram, *args, **kwargs):
        packed_input = pack_padded_sequence(spectrogram, kwargs['spectrogram_length'].cpu(),
                                            enforce_sorted=False, batch_first=True)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        res = self.fc(output)
        return res

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
