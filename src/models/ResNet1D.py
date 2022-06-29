import torch
import torch.nn as nn
import torch.nn.functional as F
import time

verbose = True
verbose = False

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_num = hidden_num
        self.output_dim = output_dim

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_list = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for l in range(self.hidden_num - 1)])
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc_in(x)
        x = self.relu(x)
        for fc in self.fc_list:
            x = fc(x)
            x = self.relu(x)
        x = self.fc_out(x)
        return x


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups)

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """

    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=self.groups)

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):

        identity = x

        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        # shortcut
        out += identity

        return out


class ResNet1D(nn.Module):
    """

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes

    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, use_bn=True,
                 use_do=True, verbose=False):
        super(ResNet1D, self).__init__()

        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap  # 2 for base model
        self.increasefilter_gap = increasefilter_gap  # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=downsample,
                use_bn=self.use_bn,
                use_do=self.use_do,
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        st = time.time()
        out = x

        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)
        # out = self.do(out)
        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)
        # out = self.softmax(out)
        if self.verbose:
            print('softmax', out.shape)

        if verbose: print(f"--- resnet forward time = {time.time() - st:.4f}")
        return out


class ResNet1D_with_hand_designed_features(nn.Module):
    """

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes

    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, use_bn=True,
                 use_do=True, verbose=False, num_hand_designed_features=2):
        super(ResNet1D_with_hand_designed_features, self).__init__()

        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap  # 2 for base model
        self.increasefilter_gap = increasefilter_gap  # 4 for base model
        self.num_hand_designed_features = num_hand_designed_features

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=downsample,
                use_bn=self.use_bn,
                use_do=self.use_do,
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels + num_hand_designed_features, n_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        hand_designed_features = x[:, :, -self.num_hand_designed_features:].squeeze(1)
        out = x[:, :, :-self.num_hand_designed_features]

        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)
            print('hand_designed_features', hand_designed_features.shape)
        # out = self.do(out)
        out = torch.cat([out, hand_designed_features], dim=1)
        if self.verbose:
            print('concatenated features', out.shape)
        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)
        # out = self.softmax(out)
        if self.verbose:
            print('softmax', out.shape)

        return out


class MyPeakDetectionCNN(nn.Module):
    """

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set largest to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes

    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, downsample_gap=2, increasefilter_gap=4, use_bn=True,
                 use_do=True, verbose=False):
        super(MyPeakDetectionCNN, self).__init__()

        self.verbose = verbose
        self.n_block = n_block
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap  # 2 for base model
        self.increasefilter_gap = increasefilter_gap  # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        self.first_block_maxpool = MyMaxPool1dPadSame(kernel_size=self.kernel_size)
        out_channels = self.base_filters

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=downsample,
                use_bn=self.use_bn,
                use_do=self.use_do,
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.dense1 = nn.Linear(128, 50)
        self.final_relu = nn.ReLU(inplace=True)
        # # self.do = nn.Dropout(p=0.5)
        self.dense2 = nn.Linear(128, 1)

    def forward(self, x):

        out = x

        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        out = self.first_block_maxpool(out)

        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)

        # final prediction
        out = out.squeeze(2)
        if self.verbose:
            print('dense1', out.shape)
        out = self.dense1(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('returning', out.shape)
        # out = self.dense2(out)

        return out





""" Dynamic spatial filtering (https://arxiv.org/abs/2105.12916) """
def soft_thresholding(x, b, a=None):
    """Remap values between [-a, b] to 0, keep the rest linear.
    """
    if a is None:
        a = b
    return (torch.clamp(x - b, min=0) * (x > 0) +
            torch.clamp(x + a, max=0) * (x <= 0))


def logm_eig(A, spd=True, upper=True):
    """Batched matrix logarithm through eigenvalue decomposition.

    Parameters
    ----------
    A : torch.Tensor
        Square matrices of shape (B, F, C, T).

    Returns
    -------
    torch.Tensor :
        Matrix logarithm of A.
    """
    # e, v = torch.symeig(A, eigenvectors=True, upper=upper)
    e, v = torch.linalg.eigh(A, UPLO="U" if upper else "L")
    e = torch.clamp(e, min=1e-10)  # clamp the eigenvalues to avoid -inf
    return v @ torch.diag_embed(
        torch.log(e), dim1=2, dim2=3) @ v.transpose(2, 3)


class SpatialFeatureExtractor(nn.Module):
    """Extract spatial features from input.
    """
    def __init__(self, kind, n_channels):
        super().__init__()
        self.kind = kind
        self.n_channels = n_channels
        self.inds = torch.triu_indices(n_channels, n_channels)

    @staticmethod
    def _cov(x):
        xm = x - x.mean(axis=3, keepdims=True)
        return xm @ xm.transpose(2, 3) / (x.shape[3] - 1)

    def forward(self, x):
        """
        x.shape = (B, F, C, T)
        """
        st = time.time()

        if self.kind == 'log_diag_cov':
            out = torch.log(torch.var(x, 3, unbiased=True))
            out[torch.isneginf(out)] = 0
        elif self.kind == 'logm_cov_eig':
            cov = self._cov(x)
            logm_cov = logm_eig(cov)
            out = logm_cov[:, :, self.inds[0], self.inds[1]]
        else:
            out = None
        if verbose: print(f"--- feature extractor time = {time.time() - st:.4f}")

        return out

    @property
    def n_outputs(self):
        if self.kind == 'log_diag_cov':
            return self.n_channels
        else:
            return int(self.n_channels * (self.n_channels + 1) / 2)


class DynamicSpatialFilter(nn.Module):
    """Dynamic spatial filter module.

    Input: (B, F, C, T) [F is the number of filters]
    Output: (B, F, C', T) [transformed input]

    Parameters
    ----------
    n_channels : int
        Number of input channel.
    mlp_input : str
        What to feed the MLP. See SpatialFeatureExtractor.
    n_hidden : int | None
        Number of hidden neurons in the MLP. If None, use `ratio`.
    ratio : float
        If `n_hidden` is None, the number of hidden neurons in the MLP is
        computed as int(ratio * n_inputs).
    n_out_channels : int | None
        Number of output ("virtual") channels in the DSF-based models (only
        affects DSF models). If None, n_out_channels = n_channels.
    apply_soft_thresholding : bool
        If True, apply soft thresholding to the spatial filter matrix W.
    return_att : bool
        If True, `forward()` returns attention values as well. Used for
        inspecting the model.
    """
    def __init__(self, n_channels, mlp_input='log_diag_cov', n_hidden=None,
                 ratio=1, n_out_channels=None, apply_soft_thresh=False,
                 return_att=False, squeeze_dim1=True):
        super().__init__()
        self.apply_soft_thresh = apply_soft_thresh
        self.return_att = return_att
        self.squeeze_dim1 = squeeze_dim1

        # Initialize spatial feature extractor
        self.feat_extractor = SpatialFeatureExtractor(
            mlp_input, n_channels)
        n_inputs = self.feat_extractor.n_outputs
        if n_hidden is None:
            n_hidden = int(ratio * n_inputs)

        # Define number of outputs
        if n_out_channels is None:
            n_out_channels = n_channels
        self.n_out_channels = n_out_channels
        n_outputs = n_out_channels * (n_channels + 1)

        self.mlp = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs)
        )

    def forward(self, x):
        st = time.time()
        input_size = x.shape
        if isinstance(x, list):  # logm was computed on CPU with transforms
            x, feats = x
            feats = feats.unsqueeze(1)
        else:
            feats = None
        if x.ndim == 3:
            b, c, _ = x.shape
            f = 1
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            b, f, c, _ = x.shape

        mlp_out = self.mlp(self.feat_extractor(x) if feats is None else feats)

        W = mlp_out[:, :, self.n_out_channels:].view(
            b, f, self.n_out_channels, c)
        if self.apply_soft_thresh:
            W = soft_thresholding(W, 0.1)
        bias = mlp_out[:, :, :self.n_out_channels].view(
            b, f, self.n_out_channels, 1)
        out = W @ x + bias
        if self.squeeze_dim1:
            out = out.squeeze(1)

        if verbose: print(f"--- dsf forward time = {time.time() - st:.4f}, input size = {input_size}")

        if self.return_att:
            return out, (W, bias)
        else:
            return out
