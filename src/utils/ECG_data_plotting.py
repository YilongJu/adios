from typing import Union, Optional, List, Tuple, Text, BinaryIO
import pathlib
import torch
import math
import warnings
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageColor
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

def Save_time_series_as_img_torch_array(signal, label=None):
    plt.rcParams["figure.figsize"] = [3.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["backend"] = "agg"
    sns.set(style="ticks", font_scale=1.33)
    plt.figure()
    if label is not None:
        if isinstance(label, str):
            legend_label = label
        else:
            if isinstance(label, torch.Tensor):
                label = label.item()
            legend_label = "JET" if label == 1 else "Sinus"
    else:
        legend_label = None

    if isinstance(label, str):
        color = cm.tab10(0)
    elif label is None:
        color = cm.tab10(0)
    else:
        color = cm.tab10(label + 1)

    plt.plot(signal, color=color, label=legend_label)
    if legend_label is not None:
        plt.legend(loc="upper right")
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    im_array_torch = torch.tensor(np.array(im))
    plt.close()
    return im_array_torch.transpose(2, 0).transpose(2, 1).unsqueeze(0)[:, :3, ...].float()


def Convert_batch_of_time_series_to_batch_of_img_torch_array(signals, labels=None, add_ind_to_legend=False):
    if labels is None:
        labels = [None] * len(signals)

    if isinstance(labels, str):
        labels = [labels] * len(signals)

    img_array_batch_list = []
    for i, (signal, label) in enumerate(zip(signals, labels)):
        # print(i, signal.shape)
        signal_reshaped = signal.squeeze(0)
        if isinstance(label, str):
            label_processed = f"{label}_{i}" if add_ind_to_legend else label
        else:
            label_processed = label
        img_array_torch_reshaped = Save_time_series_as_img_torch_array(signal_reshaped, label_processed)
        # print(img_array_torch_reshaped.shape)
        img_array_batch_list.append(img_array_torch_reshaped)

    return torch.cat(img_array_batch_list, dim=0).float()
