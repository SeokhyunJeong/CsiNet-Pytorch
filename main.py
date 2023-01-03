import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from train import train

########config#######

epochs = 100
learning_rate = 0.001
lr_decay_freq = 20
lr_decay = 0.9
print_freq = 50
device = 'cuda'


#########run########

def run(encoded_dim):
    trainer = train(epochs, encoded_dim, learning_rate, lr_decay_freq, lr_decay, print_freq, device)
    trainer.train_epoch()


dim = 16
run(encoded_dim=dim)
