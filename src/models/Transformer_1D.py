""" https://github.com/hsd1503/transformer1d """
import torch.nn as nn

class Transformer1d(nn.Module):
    """

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples, n_classes)

    Pararmetes:

    """

    def __init__(self, n_classes, n_length, num_layers, d_model, nhead, dim_feedforward, dropout, activation,
                 verbose=False, **kwargs):
        super(Transformer1d, self).__init__()

        self.d_model = d_model
        self.num_features = self.d_model
        self.nhead = nhead
        self.n_length = n_length
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.n_classes = n_classes
        self.verbose = verbose

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.dense = nn.Linear(self.d_model, self.n_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.float()
        out = x
        if self.verbose:
            print('input (n_samples, n_channel, n_length)', out.shape)
        out = out.permute(2, 0, 1)
        if self.verbose:
            print('transpose (n_length, n_samples, n_channel)', out.shape)

        out = self.transformer_encoder(out)
        if self.verbose:
            print('transformer_encoder', out.shape)

        out = out.mean(0)
        if self.verbose:
            print('global pooling', out.shape)
        #
        # out = self.dense(out)
        # if self.verbose:
        #     print('dense', out.shape)
        #
        # out = self.softmax(out)
        # if self.verbose:
        #     print('softmax', out.shape)

        return out