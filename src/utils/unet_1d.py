# Adapted from: https://github.com/applied-ai-lab/genesis.
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils.blocks as B

class UNet_1D(nn.Module):
    def __init__(self, num_blocks, img_size=64,
                 filter_start=32, in_chnls=4, out_chnls=1,
                 norm='in'):
        super(UNet_1D, self).__init__()
        # TODO(martin): make more general
        c = filter_start
        if norm == 'in':
            conv_block = B.ConvINReLU_1D
        elif norm == 'gn':
            conv_block = B.ConvGNReLU_1D
        else:
            conv_block = B.ConvReLU_1D
        if num_blocks == 4:
            enc_in = [in_chnls, c, 2 * c, 2 * c]
            enc_out = [c, 2 * c, 2 * c, 2 * c]
            dec_in = [4 * c, 4 * c, 4 * c, 2 * c]
            dec_out = [2 * c, 2 * c, c, c]
        elif num_blocks == 5:
            enc_in = [in_chnls, c, c, 2 * c, 2 * c]
            enc_out = [c, c, 2 * c, 2 * c, 2 * c]
            dec_in = [4 * c, 4 * c, 4 * c, 2 * c, 2 * c]
            dec_out = [2 * c, 2 * c, c, c, c]
        elif num_blocks == 6:
            enc_in = [in_chnls, c, c, c, 2 * c, 2 * c]
            enc_out = [c, c, c, 2 * c, 2 * c, 2 * c]
            dec_in = [4 * c, 4 * c, 4 * c, 2 * c, 2 * c, 2 * c]
            dec_out = [2 * c, 2 * c, c, c, c, c]
        elif num_blocks == 7:
            enc_in = [in_chnls, c, c, c, c, 2 * c, 2 * c]
            enc_out = [c, c, c, c, 2 * c, 2 * c, 2 * c]
            dec_in = [4 * c, 4 * c, 4 * c, 2 * c, 2 * c, 2 * c, 2 * c]
            dec_out = [2 * c, 2 * c, c, c, c, c, c]
        else:
            raise NotImplementedError("Unknown number of blocks")
        self.down = []
        self.up = []
        self.dim = 1
        # 3x3 kernels, stride 1, padding 1
        for i, o in zip(enc_in, enc_out):
            self.down.append(conv_block(i, o, 3, 1, 1))
        for i, o in zip(dec_in, dec_out):
            self.up.append(conv_block(i, o, 3, 1, 1))
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)
        self.featuremap_size = img_size // 2 ** (num_blocks - 1)
        self.mlp = nn.Sequential(
            B.Flatten(),
            nn.Linear(2 * c * self.featuremap_size ** self.dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2 * c * self.featuremap_size ** self.dim), nn.ReLU()
        )
        if out_chnls > 0:
            self.final_conv = nn.Conv1d(c, out_chnls, 1)
        else:
            self.final_conv = nn.Identity()
        self.out_chnls = out_chnls
        self.scale_factor_list = []

    def forward(self, x):
        x = x.float()
        batch_size = x.size(0)
        x_down = [x]
        skip = []
        # Down
        # print(f"x_down[-1].shape: {x_down[-1].shape}")
        for i, block in enumerate(self.down):

            act = block(x_down[-1])
            # print(f"act.shape: {act.shape}")
            act_shape_old = act.shape
            skip.append(act)
            # print(f"i = {i}, len(self.down)-1 = {len(self.down) - 1}")
            if i < len(self.down) - 1:
                act = F.interpolate(act, scale_factor=0.5, mode='nearest', recompute_scale_factor=True)
                # print(f"After interp act.shape: {act.shape}")

            self.scale_factor_list.append(act_shape_old[-1] / act.shape[-1])
            # print(f"[Down] self.scale_factor_list: {self.scale_factor_list}")

            x_down.append(act)
        # FC
        # print(f"x_down[-1].shape: {x_down[-1].shape}")
        x_up = self.mlp(x_down[-1])
        # print(f"x_up.shape: {x_up.shape}")
        x_up = x_up.view(batch_size, -1,
                         self.featuremap_size)
        self.scale_factor_list.pop()
        # Up
        # print(f"[O] self.scale_factor_list: {self.scale_factor_list}")
        for i, block in enumerate(self.up):
            # print(f"x_up.shape: {x_up.shape}")
            # print(f"skip[-1 - {i}].shape: {skip[-1 - i].shape}")
            features = torch.cat([x_up, skip[-1 - i]], dim=1)
            x_up = block(features)

            if i < len(self.up) - 1:
                scale_factor = self.scale_factor_list.pop()
                # print(f"[Up] self.scale_factor_list: {self.scale_factor_list}")
                # print(f"scale_factor: {scale_factor}")
                x_up = F.interpolate(x_up, scale_factor=scale_factor, mode='nearest', recompute_scale_factor=True)
        return self.final_conv(x_up)

class UNet_1D_separate(UNet_1D):
    """ Also returns encoder output """
    def forward(self, x):
        x = x.float()
        batch_size = x.size(0)
        x_down = [x]
        skip = []
        # Down
        # print(f"x_down[-1].shape: {x_down[-1].shape}")
        for i, block in enumerate(self.down):

            act = block(x_down[-1])
            # print(f"act.shape: {act.shape}")
            act_shape_old = act.shape
            skip.append(act)
            # print(f"i = {i}, len(self.down)-1 = {len(self.down) - 1}")
            if i < len(self.down) - 1:
                act = F.interpolate(act, scale_factor=0.5, mode='nearest', recompute_scale_factor=True)
                # print(f"After interp act.shape: {act.shape}")

            self.scale_factor_list.append(act_shape_old[-1] / act.shape[-1])
            # print(f"[Down] self.scale_factor_list: {self.scale_factor_list}")

            x_down.append(act)
        # FC
        # print(f"x_down[-1].shape: {x_down[-1].shape}")
        x_up = self.mlp(x_down[-1])
        # print(f"x_up.shape: {x_up.shape}")
        x_up = x_up.view(batch_size, -1,
                         self.featuremap_size)
        x_encoded = x_up.copy()
        self.scale_factor_list.pop()
        # Up
        # print(f"[O] self.scale_factor_list: {self.scale_factor_list}")
        for i, block in enumerate(self.up):
            # print(f"x_up.shape: {x_up.shape}")
            # print(f"skip[-1 - {i}].shape: {skip[-1 - i].shape}")
            features = torch.cat([x_up, skip[-1 - i]], dim=1)
            x_up = block(features)

            if i < len(self.up) - 1:
                scale_factor = self.scale_factor_list.pop()
                # print(f"[Up] self.scale_factor_list: {self.scale_factor_list}")
                # print(f"scale_factor: {scale_factor}")
                x_up = F.interpolate(x_up, scale_factor=scale_factor, mode='nearest', recompute_scale_factor=True)
        return self.final_conv(x_up), x_encoded

