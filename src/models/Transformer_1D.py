# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:20:16 2021

@author: bjorn

script for transformer model
based on: https://github.com/bh1995/AF-classification,
which is based on: [8] A. Natarajan, et al.. A Wide and Deep Transformer Neural Network 
for 12-Lead ECG Classification. 2020 Computing in Cardiology, 2020, pp. 1–4, doi: 10.22489/CinC.2020.107.
"""
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # max_len x d_model
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (
                    -9.210340371976184 / d_model))  # 9.210340371976184 = math.log(10000.0)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # max_len x 1 x d_model
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


class Transformer1D(nn.Module):

    def __init__(self, d_model=64, nhead=1, dim_feedforward=128, nlayers=3, n_length=300, embedding_dim=64,
                 n_conv_layers=2, n_class=2, dropout=0.5, dropout_other=0.1, use_raw_patch=False, **kwargs):
        super(Transformer1D, self).__init__()
        self.model_type = 'Transformer'
        self.n_class = n_class
        self.n_conv_layers = n_conv_layers
        self.num_features = embedding_dim
        self.relu = torch.nn.ReLU()
        self.d_model = d_model
        self.d_size = n_length // d_model
        self.use_raw_patch = use_raw_patch
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=n_length)
        #         self.pos_encoder2 = PositionalEncoding(6, dropout)
        self.self_att_pool = SelfAttentionPooling(d_model)
        #         self.self_att_pool2 = SelfAttentionPooling(d_model)
        encoder_layers = TransformerEncoderLayer(d_model=d_model,
                                                 nhead=nhead,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout
                                                 )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #         self.transformer_encoder2 = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.flatten_layer = torch.nn.Flatten()
        # Define linear output layers
        if n_class == 2:
            self.decoder = nn.Sequential(nn.Linear(d_model, d_model),
                                         nn.Dropout(dropout_other),
                                         nn.Linear(d_model, d_model),
                                         nn.Linear(d_model, embedding_dim))
        # else:
        #   self.decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(0.1),
        #                                nn.Linear(d_model, d_model), nn.Dropout(0.1),
        #                                nn.Linear(d_model, n_class))
        #         if n_class == 2:
        #             self.decoder2 = nn.Sequential(nn.Linear(d_model, d_model),
        #                                        nn.Dropout(dropout_other),
        #                                       #  nn.Linear(d_model, d_model),
        #                                        nn.Linear(d_model, 64))
        # Linear output layer after concat.
        #         self.fc_out1 = torch.nn.Linear(64+64, 64)
        self.fc_out1 = torch.nn.Linear(embedding_dim, embedding_dim)  # [Yilong 20221108] Removed the RRI input
        #         self.fc_out2 = torch.nn.Linear(embedding_dim, 2) # if two classes problem is binary
        # self.init_weights()
        # Transformer Conv. layers
        if not self.use_raw_patch:
            self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=2 * embedding_dim, kernel_size=3, stride=1,
                                         padding='same')
            self.conv2 = torch.nn.Conv1d(in_channels=2 * embedding_dim, out_channels=d_model, kernel_size=3, stride=1,
                                         padding=1)
            self.conv = torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding='same')
            # self.bn1 = nn.BatchNorm1d(128)
            # self.bn2 = nn.BatchNorm1d(d_model)
            self.maxpool_list = [
                torch.nn.MaxPool1d(kernel_size=2), # was 4
                torch.nn.MaxPool1d(kernel_size=2) # was 5
            ]

        self.dropout = torch.nn.Dropout(p=0.1)
        # self.avg_maxpool = nn.AdaptiveAvgPool2d((64, 64))
        # RRI layers

    #         self.conv1_rri = torch.nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3)
    #         self.conv2_rri = torch.nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=3)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = src.float()
        if self.use_raw_patch:
            # print(f"src.shape = {src.shape}, self.d_size = {self.d_size}")
            src = src.view(-1, self.d_size, self.d_model)
            # print('src shape after reshape:', src.shape)
        else:
            # src = self.encoder(src) * math.sqrt(self.d_model)
            # size input: [batch, sequence, embedding dim.]
            # src = self.pos_encoder(src)
            # print('initial src shape:', src.shape)
            #         src = src.view(-1, 1, src.shape[1]) # Resize to --> [batch, input_channels, signal_length]
            src = self.relu(self.conv1(src))
            # print('src shape after conv1:', src.shape)
            src = self.relu(self.conv2(src))
            # src = self.maxpool(self.relu(src))
            # print('src shape after conv2:', src.shape)
            for i in range(self.n_conv_layers):
                src = self.relu(self.conv(src))
                src = self.maxpool_list[i](src)

        # print('src shape after more convlayers:', src.shape)
        # src = self.maxpool(self.relu(src))
        src = src.permute(2, 0, 1)  # reshape from [batch, embedding dim., sequence] --> [sequence, batch, embedding dim.]
        # print('src shape after permute:', src.shape)
        src = self.pos_encoder(src)
        # print('src shape after copos_encodernv1:', src.shape)  # [batch, embedding, sequence]
        output = self.transformer_encoder(src)  # output: [sequence, batch, embedding dim.], (ex. [3000, 5, 512])
        # print('output shape 1:', output.shape)
        # output = self.avg_maxpool(output)
        # output = torch.mean(output, dim=0) # take mean of sequence dim., output: [batch, embedding dim.]
        output = output.permute(1, 0, 2)
        output = self.self_att_pool(output)
        # print('output shape 2:', output.shape)
        logits = self.decoder(output)  # output: [batch, n_class]
        # print('output shape 3:', logits.shape)
        logits_concat = logits
        # Linear output layer after concat.
        xc = self.flatten_layer(logits_concat)
        #         print('shape after flatten', xc.shape)
        #         xc = self.fc_out2(self.dropout(self.relu(self.fc_out1(xc))))
        xc = self.dropout(self.relu(self.fc_out1(xc)))

        return xc