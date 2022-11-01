import numpy as np
import torch

from src.transforms import warp_ops


def Add_Gaussian_noise(x, dataset_name="ecg-TCH-40_patient-20220201", mean=0, std=-1):
    mult_factor = 1
    if dataset_name in ['ptb', 'physionet2020']:
        # The ECG frames were normalized in amplitude between the values of 0 and 1.
        variance_factor = 0.01 * mult_factor
    elif dataset_name in ['cardiology', 'chapman']:
        variance_factor = 10 * mult_factor
    elif dataset_name in ['physionet', 'physionet2017']:
        variance_factor = 100 * mult_factor
    elif dataset_name in ["ecg-TCH-40_patient-20220201", "ecg-TCH-40_patient-20220201_with_CVP"]:
        variance_factor = 0.01 * mult_factor
    else:
        raise NotImplementedError("Dataset not implemented")

    if std > 0:
        variance_factor = std

    gauss_noise = np.random.normal(mean, variance_factor, size=x.shape)
    return x + gauss_noise


def Flip_Along_Y(x):
    return np.flip(x)


def Flip_Along_X(x):
    return -x


def Transverse_transformation(x):
    """ Another way of baseline wander """
    upper = 1.5
    lower = 0.5
    t_linspace = np.linspace(0, 2 * np.pi, len(x))
    period = 2 * np.pi / np.random.uniform(0.5, 4)
    phase = np.random.uniform(0, 2 * np.pi)
    mask = (upper - lower) / 2 * np.sin(t_linspace / period + phase) + (upper + lower) / 2
    return x * mask


""" https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array """


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def Longitudinal_transformation(x):
    """ Another way of temporal warping """
    amplitude = 20
    t_linspace = np.linspace(0, 2 * np.pi, 300)
    period = 2 * np.pi / np.random.uniform(8, 12)
    phase = np.random.uniform(0, 2 * np.pi)
    mask = amplitude * np.sin(t_linspace / period + phase)
    signal_aug = np.zeros_like(x) * np.nan
    for i, v in enumerate(x):
        new_i = i + np.round(mask[i]).astype(int)
        if new_i >= 0 and new_i < len(signal_aug):
            signal_aug[new_i] = v

    signal_aug_interp = signal_aug
    nans, x = nan_helper(signal_aug_interp)
    signal_aug_interp[nans] = np.interp(x(nans), x(~nans), signal_aug_interp[~nans])
    return signal_aug_interp


""" ======= Below this line ====== 
    Adapted from the TaskAug paper 
    =============================== """
def _Temporal_Warp(x, mag=1., warp_obj=None):
    if warp_obj is None:
        warp_obj = warp_ops.RandWarpAug([len(x)])
    mag = 100 * (mag ** 2)
    x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    return warp_obj(x, mag)


def _Baseline_wander(x, mag=0.):
    x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    BS, C, L = x.shape

    # form baseline drift
    strength = 0.25 * torch.sigmoid(torch.tensor(mag)) * (torch.rand(BS).to(x.device).view(BS, 1, 1))
    strength = strength.view(BS, 1, 1)

    frequency = ((torch.rand(BS) * 20 + 10) * 10 / 60).view(BS, 1, 1)  # typical breaths per second for an adult
    phase = (torch.rand(BS) * 2 * np.pi).view(BS, 1, 1)
    drift = strength * torch.sin(torch.linspace(0, 1, L).view(1, 1, -1) * frequency.float() + phase.float()).to(
        x.device)
    return x + drift


def _Gau_noise(x, mag=0.):  # TaskAug
    x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    BS, C, L = x.shape
    stdval = torch.std(x, dim=2).view(BS, C, 1).detach()
    noise = 0.25 * stdval * torch.sigmoid(torch.tensor(mag)) * torch.randn(BS, C, L).to(x.device)
    return x + noise


def _Magnitude_scale(x, mag=0.):
    x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    BS, C, L = x.shape
    strength = torch.sigmoid(torch.tensor(mag)) * (-0.5 * (torch.rand(BS).to(x.device)).view(BS, 1, 1) + 1.25)
    strength = strength.view(BS, 1, 1)
    return x * strength


def _Time_mask(x, mag=0.1):
    x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    x_aug = x.clone()
    # get shapes
    BS, C, L = x.shape

    nmf = int(mag * L)
    start = torch.randint(0, L - nmf, [1]).long()
    end = (start + nmf).long()
    x_aug[:, :, start:end] = 0.
    return x_aug

def _Random_temporal_displacement(x, mag=0.5, warp_obj=None):
    if warp_obj is None:
        warp_obj = warp_ops.DispAug([len(x)])
    disp_mag = 100 * (mag ** 2)
    x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    return warp_obj(x, disp_mag)

def Temporal_Warp(x, mag=1.):
    return _Temporal_Warp(x, mag=mag).squeeze().numpy()

def Baseline_wander(x, mag=0.):
    return _Baseline_wander(x, mag=mag).squeeze().numpy()

def Gau_noise(x, mag=0.):
    return _Gau_noise(x, mag=mag).squeeze().numpy()

def Magnitude_scale(x, mag=0.):
    return _Magnitude_scale(x, mag=mag).squeeze().numpy()

def Time_mask(x, mag=0.1):
    return _Time_mask(x, mag=mag).squeeze().numpy()

def Random_temporal_displacement(x, mag=0.5):
    return _Random_temporal_displacement(x, mag=mag).squeeze().numpy()