import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from src.models.Debugging import *


class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8, **kwargs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
            int((configs.seq_len - patch_len) / stride + 2)

        print(f"Using config: {configs}")
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = Flatten_Head(configs.enc_in, self.head_nf, configs.pred_len,
                                     head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = Flatten_Head(configs.enc_in, self.head_nf, configs.seq_len,
                                     head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.head_nf = 150 / (configs.enc_in / 128) * (configs.d_model / 64)
            print(f"self.head_nf = {self.head_nf}")

            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.fc = nn.Linear(
                int(self.head_nf * configs.enc_in), configs.num_class)

        self.num_features = int(self.head_nf * configs.enc_in)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        # print(f"[After normalization] x_enc.shape = {x_enc.shape}")

        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        # print(f"[Before projection] output.shape = {output.shape}")
        output = self.fc(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x_enc = x_enc.float()
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="classification")
    parser.add_argument("--seq_len", type=int, default=300)
    parser.add_argument("--patch_len", type=int, default=15)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--output_attention", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--n_heads", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--enc_in", type=int, default=128)
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--e_layers", type=int, default=4)
    configs = parser.parse_args()

    model = PatchTST(configs, configs.patch_len, configs.stride)
    batch_size = 10
    input_size = (batch_size, 1, 300)  # without the batch dimension
    print(summary(model, input_size=input_size, verbose=1))
    print(model.fc)