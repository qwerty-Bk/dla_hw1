from torch import nn, transpose
from hw_asr.base import BaseModel


class ConvBlock(nn.Module):
    """convolution + batch norm"""
    def __init__(self, c_in, c_out, kernel_size, tcs, stride, dilation, padding):
        super(ConvBlock, self).__init__()
        if tcs:
            self.seq = nn.Sequential(
                nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=kernel_size,
                          stride=stride, dilation=dilation, padding=padding, groups=c_in, bias=False),
                nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm1d(c_out, eps=1e-3, momentum=0.1)
            )
        else:
            self.seq = nn.Sequential(
                nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, bias=False),
                nn.BatchNorm1d(c_out, eps=1e-3, momentum=0.1)
            )

    def forward(self, x):
        return self.seq(x)


class MainBlock(nn.Module):
    """B_i in the original paper"""
    def __init__(self, c_in, c_out, r, kernel_size, stride, dilation):
        super(MainBlock, self).__init__()
        seq = []
        padding = kernel_size // 2 * dilation
        c_now = c_in
        for i in range(r - 1):
            seq.append(ConvBlock(c_now, c_out, kernel_size, True, stride, dilation, padding))
            seq.append(nn.ReLU())
            c_now = c_out
        seq.append(ConvBlock(c_now, c_out, kernel_size, True, stride, dilation, padding))
        self.seq = nn.Sequential(*seq)
        self.residual = ConvBlock(c_in, c_out, kernel_size=1, tcs=False, stride=1, dilation=1, padding=0)
        self.activation = nn.ReLU()

    def forward(self, x):
        output = self.seq(x)
        output = output + self.residual(x)
        return self.activation(output)


class SimpleBlock(nn.Module):
    """convolution + batch norm"""
    def __init__(self, c_in, c_out, kernel_size, tcs, stride, dilation, padding=None):
        super(SimpleBlock, self).__init__()

        if padding is None:
            padding = kernel_size // 2
        self.seq = nn.Sequential(
            ConvBlock(c_in, c_out, kernel_size, tcs, stride, dilation, padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)


class QuartzNetModel(BaseModel):
    def __init__(self, n_feats, n_class, r=5, *args, **kwargs):
        super(QuartzNetModel, self).__init__(n_feats, n_class, *args, **kwargs)
        print("Building QuartzNet-{}x{}".format(r * 5, 5))

        kernel_sizes_b = [33, 39, 51, 63, 75]
        channels_b = [256, 256, 256, 512, 512, 512]

        self.c1 = SimpleBlock(n_feats, 256, 33, True, stride=2, dilation=1)

        self.b = []
        for i in range(5):
            self.b.append(
                MainBlock(channels_b[i], channels_b[i + 1], r, kernel_sizes_b[i], 1, 1)
            )
        self.b = nn.Sequential(*self.b)

        self.c2 = SimpleBlock(channels_b[-1], 512, 87, True, stride=1, dilation=1, padding=86)
        self.c3 = SimpleBlock(512, 1024, 1, True, stride=1, dilation=1)
        self.c4 = nn.Conv1d(1024, n_class, kernel_size=1, stride=1, dilation=2, bias=False)

    def forward(self, spectrogram, *args, **kwargs):
        output = transpose(spectrogram, 1, 2)
        output = self.c1(output)
        output = self.b(output)
        output = self.c2(output)
        output = self.c3(output)
        output = self.c4(output)
        return transpose(output, 1, 2)

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
