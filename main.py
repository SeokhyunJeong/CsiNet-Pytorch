import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from train import train

########config#######

epochs = 40
learning_rate = 0.001
lr_decay_freq = 20
lr_decay = 0.9
print_freq = 100
device = 'cuda'


#########run########

def run(encoded_dim):
    trainer = train('./filepath/data_large.csv',
                    epochs, encoded_dim, learning_rate, lr_decay_freq, lr_decay, print_freq, device)
    encoder, decoder = trainer.train_epoch()
    online_trainer1 = train('./filepath/data_online.csv',
                            0, encoded_dim, learning_rate, lr_decay_freq, lr_decay, print_freq, device,
                            encoder=encoder, decoder=decoder)
    online_trainer1.train_epoch()
    online_trainer2 = train('./filepath/data_online.csv',
                            epochs, encoded_dim, learning_rate, lr_decay_freq, lr_decay, print_freq, device,
                            encoder=encoder, decoder=decoder)
    online_trainer2.train_online_epoch()
    online_trainer3 = train('./filepath/data_online.csv',
                            epochs, encoded_dim, learning_rate, lr_decay_freq, lr_decay, print_freq, device)
    online_trainer3.train_epoch()


dim = 128
run(encoded_dim=dim)
