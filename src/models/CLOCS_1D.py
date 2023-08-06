#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:14:18 2020
https://github.com/danikiyasseh/CLOCS
@author: Dani Kiyasseh
"""

import torch.nn as nn
import torch
from src.models.Debugging import *

# %%
""" Functions in this scripts:
    1) cnn_network_contrastive
    2) second_cnn_network
"""

# %%




# num_classes = 3

class cnn_network_contrastive(nn.Module):
    """ CNN for Self-Supervision """

    def __init__(self, dropout_type="drop1d", p1=0.1, p2=0.1, p3=0.1, nencoders=1, stride=3, n_classes=2, in_channels=None,
                 c4_multiplier=10, embedding_dim=256, kernel_size=7, trial='CLOCS', device='', in_channels_type=None, **kwargs):
        # dropout_type = ['drop1d'] or 'drop2d'
        # p1 = dropout probability for first layer (default = 0.1)
        # p2 = dropout probability for second layer (default = 0.1)
        # p3 = dropout probability for third layer (default = 0.1)
        # nencoders = number of encoders (default = 1)
        # embedding_dim = dimension of latent embedding (default = 256)
        # trial = ['CMC'] or 'CLOCS' (default = 'CLOCS')
        # device = ['cpu'] or 'cuda' (default = 'cpu')
        c1 = 1  # b/c single time-series
        c2 = 4  # 4
        c3 = 16  # 16
        c4 = 32  # 32
        k = kernel_size  # kernel size #7
        # s = 3  # stride #3
        if k in [5, 6, 7]:
            c4_multiplier = 3
        elif k in [3, 4]:
            c4_multiplier = 4


        s = stride
        print(s, p1, p2, p3, nencoders, embedding_dim, trial, device)
        print(f"Using kernel size: {k}")

        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.num_features = self.embedding_dim
        if in_channels_type is not None:
            in_channels = len(in_channels_type.split(','))
        if in_channels is not None:
            c1 = in_channels
        self.in_channels = in_channels
        self.c4_multiplier = c4_multiplier

        if dropout_type == 'drop1d':
            self.dropout1 = nn.Dropout(p=p1)  # 0.2 drops pixels following a Bernoulli
            self.dropout2 = nn.Dropout(p=p2)  # 0.2
            self.dropout3 = nn.Dropout(p=p3)
        elif dropout_type == 'drop2d':
            self.dropout1 = nn.Dropout2d(p=p1)  # drops channels following a Bernoulli
            self.dropout2 = nn.Dropout2d(p=p2)
            self.dropout3 = nn.Dropout2d(p=p3)

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.maxpool = nn.MaxPool1d(2)
        self.trial = trial
        # self.device = device
        self.view_modules = nn.ModuleList()
        self.view_linear_modules = nn.ModuleList()
        for n in range(nencoders):
            self.view_modules.append(nn.Sequential(
                nn.Conv1d(c1, c2, k, s),
                nn.BatchNorm1d(c2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                self.dropout1,
                nn.Conv1d(c2, c3, k, s),
                nn.BatchNorm1d(c3),
                nn.ReLU(),
                nn.MaxPool1d(2),
                self.dropout2,
                nn.Conv1d(c3, c4, k, s),
                nn.BatchNorm1d(c4),
                nn.ReLU(),
                nn.MaxPool1d(2),
                self.dropout3
            ))
            self.view_linear_modules.append(nn.Linear(c4 * c4_multiplier, self.embedding_dim))

        print(f"c1, c2, k, s, c3, c4, c4_multiplier, embedding_dim: {c1, c2, k, s, c3, c4, c4_multiplier, self.embedding_dim}")

    def forward(self, x):
        # Raw input shape: (BxCxS) = (batch_size x num_channels x num_samples)
        """ Forward Pass on Batch of Inputs - CLOCS
        Args:
            x (torch.Tensor): inputs with N views (BxSxN)
        Outputs:
            h (torch.Tensor): latent embedding for each of the N views (BxHxN)
        """
        # print(f"clocs 1d input dim: {x.shape}")
        x = x.float()
        x = x.unsqueeze(3)  # [20221009] Each view is fed seperatedly to the CNN
        batch_size = x.shape[0]
        # nsamples = x.shape[2]
        nviews = x.shape[3]
        # nviews = 1
        latent_embeddings = torch.empty(batch_size, self.embedding_dim, nviews, device=self.device)
        for n in range(nviews):
            """ Obtain Inputs From Each View """
            h = x[:, :, :, n]

            if self.trial == 'CMC':
                h = self.view_modules[n](h)  # nencoders = nviews
                h = torch.reshape(h, (h.shape[0], h.shape[1] * h.shape[2]))
                h = self.view_linear_modules[n](h)
            else:
                # print(f"[before view_modules] h.shape: {h.shape}")
                h = self.view_modules[0](h)  # nencoder = 1 (used for all views)
                # print(f"[before reshape] h.shape: {h.shape}")
                h = torch.reshape(h, (h.shape[0], h.shape[1] * h.shape[2]))
                # print(f"[after reshape] h.shape: {h.shape}")
                h = self.view_linear_modules[0](h)

            latent_embeddings[:, :, n] = h

        # latent_embeddings.shape = (batch_size x embedding_dim x nviews)
        latent_embeddings = latent_embeddings.squeeze(2)
        return latent_embeddings


class second_cnn_network(nn.Module):

    def __init__(self, first_model, noutputs, embedding_dim=256):
        super(second_cnn_network, self).__init__()
        self.first_model = first_model
        self.linear = nn.Linear(embedding_dim, noutputs)

    def forward(self, x):
        x = x.float()
        h = self.first_model(x)
        h = h.squeeze()  # to get rid of final dimension from torch.empty before
        output = self.linear(h)
        return output

if __name__ == "__main__":
    model = cnn_network_contrastive(stride=2, embedding_dim=32, kernel_size=3)
    input_dim = (5, 1, 300) # without the batch dimension
    print(summary(model, input_dim))
