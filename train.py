import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data import load_data
from model import Encoder, Decoder


import random
import os
import time

gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

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


class train(nn.Module):
    def __init__(self,
                 file_path,
                 epochs,
                 encoded_dim,
                 learning_rate,
                 lr_decay_freq,
                 lr_decay,
                 print_freq,
                 device,
                 online=False,
                 encoder=None,
                 decoder=None,
                 ):
        super().__init__()
        self.epochs = epochs
        self.encoded_dim = encoded_dim
        self.learning_rate = learning_rate
        self.lr_decay_freq = lr_decay_freq
        self.lr_decay = lr_decay
        self.print_freq = print_freq
        self.online = online

        #### 1.load data ####
        self.train_loader, self.test_loader = load_data(file_path)

        self.encoder_ue = Encoder(encoded_dim).to(device)
        self.decoder_bs = Decoder(encoded_dim).to(device)
        if encoder is not None:
            self.encoder_ue.load_state_dict(encoder)
        if decoder is not None:
            self.decoder_bs.load_state_dict(decoder)

        self.criterion = nn.MSELoss()
        self.optimizer_ue = optim.Adam(self.encoder_ue.parameters())
        self.optimizer_bs = optim.Adam(self.decoder_bs.parameters())
        SEED = 42
        seed_everything(SEED)

    def train_epoch(self):

        self.encoder_ue.train()
        self.decoder_bs.train()

        #### 2. train_epoch ####
        for epoch in range(self.epochs):

            if epoch % self.lr_decay_freq == 0 and epoch > 0:
                self.optimizer_ue.param_groups[0]['lr'] = self.optimizer_ue.param_groups[0]['lr'] * self.lr_decay
                self.optimizer_bs.param_groups[0]['lr'] = self.optimizer_bs.param_groups[0]['lr'] * self.lr_decay

#             __________                           __________
#             |         \                         /         |
#             |          |                       |          |
# input ----> |encoder_ue| ----> codeword ---->  |decoder_bs| ----> output ----------
# (2,Nc,Nt)   |          |     (encoded_dim)     |          |      (2,Nc,Nt)        |
#   |         |_________/                         \_________|                       v
#   |                                                                              MSE
#   |                                                                               ^
#   |_______________________________________________________________________________|

            for i, input in enumerate(self.train_loader):
                input = input.cuda()
                codeword = self.encoder_ue(input)
                output = self.decoder_bs(codeword)

                loss = self.criterion(output, input)
                loss.backward()
                self.optimizer_ue.step()
                self.optimizer_ue.zero_grad()
                self.optimizer_bs.step()
                self.optimizer_bs.zero_grad()

                if i % self.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss {loss:.4f}\t'.format(
                        epoch, i, len(self.train_loader), loss=loss.item()))

            #### 3. validate ####
        self.encoder_ue.eval()
        self.decoder_bs.eval()

        total_loss = 0
        start = time.time()
        with torch.no_grad():
            for i, input in enumerate(self.test_loader):
                input = input.cuda()
                codeword = self.encoder_ue(input, test=True)
                output = self.decoder_bs(codeword, test=True)
                total_loss += self.criterion(output, input).item()

            end = time.time()
            t = end - start
            average_loss = total_loss / len(list(enumerate(self.test_loader)))
            print('NMSE %.6ftime %.3f' % (average_loss, t))

        channel_visualization(input.detach().cpu().numpy()[12][0])
        channel_visualization(output.detach().cpu().numpy()[12][0])

        torch.save(self.encoder_ue.state_dict(), './trained_models/encoder_ue_pretrain.pt')
        torch.save(self.decoder_bs.state_dict(), './trained_models/decoder_bs_pretrain.pt')

        return self.encoder_ue.state_dict(), self.decoder_bs.state_dict()

    def train_online_epoch(self):

        self.encoder_ue.eval()
        self.decoder_bs.train()

        #### 2. train_epoch ####
        for epoch in range(self.epochs):

            if epoch % self.lr_decay_freq == 0 and epoch > 0:
                self.optimizer_ue.param_groups[0]['lr'] = self.optimizer_ue.param_groups[0]['lr'] * self.lr_decay
                self.optimizer_bs.param_groups[0]['lr'] = self.optimizer_bs.param_groups[0]['lr'] * self.lr_decay

#             __________                           __________                     __________
#             |         \                         /         |                     |         \
#             |          |                       |          |                     |          |
# input ----> |encoder_ue| ----> codeword ---->  |decoder_bs| ----> output  ----> |encoder_ue| ----> estimated_codeword
# (2,Nc,Nt)   |          |     (encoded_dim)     |          |      (2,Nc,Nt)      |          |          (encoded_dim)
#             |_________/           |             \_________|                     |_________/             |
#              cannot be            |               can be                         cannot be              V
#               trained             |              trained                          trained              MSE
#                                   |                                                                     ^
#                                   |_____________________________________________________________________|

            for i, input in enumerate(self.train_loader):
                input = input.cuda()
                codeword = self.encoder_ue(input)
                output = self.decoder_bs(codeword)
                estimated_codeword = self.encoder_ue(output)

                loss = self.criterion(estimated_codeword, codeword)
                loss.backward()
                self.optimizer_bs.step()
                self.optimizer_bs.zero_grad()

                if i % self.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss {loss:.4f}\t'.format(
                        epoch, i, len(self.train_loader), loss=loss.item()))

            #### 3. validate ####
        self.encoder_ue.eval()
        self.decoder_bs.eval()

        total_loss = 0
        start = time.time()
        with torch.no_grad():
            for i, input in enumerate(self.test_loader):
                input = input.cuda()
                codeword = self.encoder_ue(input, test=True)
                output = self.decoder_bs(codeword, test=True)
                total_loss += self.criterion(output, input).item()

            end = time.time()
            t = end - start
            average_loss = total_loss / len(list(enumerate(self.test_loader)))
            print('NMSE %.6ftime %.3f' % (average_loss, t))

        channel_visualization(input.detach().cpu().numpy()[12][0])
        channel_visualization(output.detach().cpu().numpy()[12][0])

        torch.save(self.encoder_ue.state_dict(), './trained_models/encoder_ue_pretrain.pt')
        torch.save(self.decoder_bs.state_dict(), './trained_models/decoder_bs_pretrain.pt')

        return self.encoder_ue.state_dict(), self.decoder_bs.state_dict()

#### 4. show results ####
