import random

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

batch_size = 16
Nc = 32  # The number of subcarriers
Nt = 32  # The number of transmit antennas
N_channel = 2  # Real, Imaginary
# encoded_dim = 32  # dimension of the codeword
# revise 20230525: erase useless constant variable 'encoded_dim' declared in this file.

def channel_visualization(image):
    fig, ax = plt.subplots()
    plot = ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest', origin='upper')
    plt.colorbar(plot)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.show()


def show(image_torch):
    image = image_torch[4][0].detach().cpu().numpy()
    channel_visualization(image)


class Encoder(nn.Module):
    # input: (batch_size, Nc, Nt) channel matrix
    # output: (batch_size, encoded_dim) codeword
    # CSI_NET
    def __init__(self, encoded_dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(num_features=2),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.fc = nn.Linear(in_features=2 * Nc * Nt, out_features=encoded_dim)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, test=False):
        out = self.conv_block(x)
        # out.shape = (batch_size, 2, Nc, Nt)
        # if test: show(out)
        out = torch.reshape(out, (batch_size, -1))
        # out.shape = (batch_size, 2*Nc*Nt)
        out = self.fc(out)
        # if test: show(torch.reshape(out, (batch_size, 1, 4, encoded_dim//4)))
        # out.shape = (batch_size, encoded_dim)

        return out


class Refinenet(nn.Module):
    # input: (batch_size, 2, Nc, Nt)
    # output: (batch_size, 2, Nc, Nt)
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 2, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(num_features=2),
            nn.LeakyReLU(negative_slope=0.3),
        )

    def forward(self, x):
        skip_connection = x
        out = self.conv1(x)
        # out.shape = (batch_size, 8, Nc, Nt)
        out = self.conv2(out)
        # out.shape = (batch_size, 16, Nc, Nt)
        out = self.conv3(out)
        # out.shape = (batch_size, 2, Nc, Nt)
        out = out + skip_connection

        return out


class Decoder(nn.Module):
    # input: (batch_size, encoded_dim) codeword
    # output: (batch_size, Nc, Nt) reconstructed channel matrix
    # CSI_NET
    def __init__(self, encoded_dim, test=False):
        super().__init__()
        self.fc = nn.Linear(in_features=encoded_dim, out_features=2 * Nc * Nt)
        self.refine1 = Refinenet()
        self.refine2 = Refinenet()
        self.test = test

    def forward(self, x, test=False):
        # x.shape = (batch_size, encoded_dim)
        out = self.fc(x)
        # out.shape = (batch_size, 2*Nc*Nt)
        out = torch.reshape(out, (batch_size, 2, Nc, Nt))
        # if test: show(out)
        # out.shape = (batch_size, 2, Nc, Nt)
        out = self.refine1(out)
        # if test: show(out)
        # out.shape = (batch_size, 2, Nc, Nt)
        out = self.refine2(out)
        # if test: show(out)
        # out.shape = (batch_size, 2, Nc, Nt)

        # channel_real = out[:, 0, :, :]
        # channel_imag = out[:, 1, :, :]
        # out = channel_real + 1j * channel_imag
        # out.shape = (batch_size, Nc, Nt)
        return out

class SeriesAdditionalBlock(nn.Module):
    # input: (batch_size, 2, Nc, Nt) initial reconstructed CSI
    # output: (batch_size, 2, Nc, Nt) final reconstructed CSI
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=2 * Nc * Nt, out_features=2 * Nc * Nt),
            nn.BatchNorm1d(num_features=2 * Nc * Nt),
            nn.LeakyReLU(),
            nn.Linear(in_features=2 * Nc * Nt, out_features=2 * Nc * Nt),
            nn.BatchNorm1d(num_features=2 * Nc * Nt),
            nn.LeakyReLU(),
            nn.Linear(in_features=2 * Nc * Nt, out_features=2 * Nc * Nt),
        )

    def forward(self, x, test=False):
        x = torch.reshape(x, (batch_size, -1))
        out = self.fc(x)
        out = out + x  # skip connection
        out = torch.reshape(out, (batch_size, 2, Nc, Nt))
        return out
   
