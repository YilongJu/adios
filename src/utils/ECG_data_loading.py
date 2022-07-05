import os
import platform
import time
import pickle
import numpy as np
import pandas as pd
import argparse
import random
import torch
from torch.utils.data import Dataset

patient_ID_list_train = [398573, 462229, 637891, 667681, 537854, 628521, 642321, 662493,
                         387479, 624179, 417349, 551554, 631270, 655769, 678877] # 15
patient_ID_list_test = [756172,  424072, 748555, 748900, 759678, 741235, 595561, 678607,
                        782501, 510915, 771495, 740475, 533362, 581650, 803389, 577874,
                        681150, 536886, 477589, 844864, 824744, 515544, 771958, 725860, 609090] # 25
patient_ID_list_val = [462229, 642321, 387479]
patient_ID_list_dev = [patient_ID for patient_ID in patient_ID_list_train if patient_ID not in patient_ID_list_val]


def Data_preprocessing(args):
    """ Preprocessing """
    data_folder = os.path.normpath("/mnt/scratch07/yilong") if args.cluster_name in ["b1", "b3", "b4"] else os.path.normpath("/mnt/group1/yilong/JET-Detection-Data")
    data_folder_2 = data_folder
    large_data_folder = data_folder

    save_folder = os.path.join(data_folder, "Results")

    if platform.system() == "Darwin":
        print("Using MacOS.")
        data_folder = os.path.normpath("/Users/yj31/Dropbox/Study/GitHub/JET-Detection")
        data_folder_2 = data_folder
    elif platform.system() == "Linux":
        print("Using Linux.")
    else:
        print("Using Windows.")
        data_folder = os.path.normpath("")
        data_folder_2 = os.path.normpath("D:\\Backup\\JET-Detection\\")
        large_data_folder = os.path.normpath("D:\\Backup\\JET-Detection\\Heartbeats_dict_20220201\\")

    # TODO: modify this step to make it use less memory at once
    if args.read_data_by_chunk:
        data_chunk_list = []
        for data_filename in os.listdir(os.path.join(data_folder, args.data_chunk_folder)):
            data_chunk_list.append(pd.read_csv(os.path.join(data_folder, args.data_chunk_folder, data_filename)))
        feature_df_all_selected_with_ecg = pd.concat(data_chunk_list, axis=0)
    else:
        feature_df_all_selected_with_ecg = pd.read_csv(os.path.join(data_folder_2, "feature_df_all_selected_with_ecg_20220210_rtfixed.csv"))
    # feature_df_all_selected_with_ecg = pd.read_csv(os.path.join(data_folder_2, "feature_df_all_selected_with_ecg_20220210_rtfixed_sample10000.csv"))
    feature_with_ecg_df_train = feature_df_all_selected_with_ecg.query(f"patient_ID in {patient_ID_list_train}")
    feature_with_ecg_df_test = feature_df_all_selected_with_ecg.query(f"patient_ID in {patient_ID_list_test}")
    feature_with_ecg_df_dev = feature_df_all_selected_with_ecg.query(f"patient_ID in {patient_ID_list_dev}")
    feature_with_ecg_df_val = feature_df_all_selected_with_ecg.query(f"patient_ID in {patient_ID_list_val}")
    print(f"Data shape: {feature_df_all_selected_with_ecg.shape}, train: {feature_with_ecg_df_train.shape}, test: {feature_with_ecg_df_test.shape}")

    # return feature_with_ecg_df_dev, feature_with_ecg_df_val, save_folder
    return feature_with_ecg_df_train, feature_with_ecg_df_test, save_folder


def Get_exp_name(args):
    channel_ID = args.channel_ID
    seed = args.seed
    use_simulator = args.use_simulator

    n_classes = 2
    base_filters = args.base_filters
    kernel_size = args.kernel_size
    stride = args.stride
    n_block = args.n_block
    groups = args.groups
    downsample_gap = args.downsample_gap
    increasefilter_gap = args.increasefilter_gap

    batch_size = args.batch_size
    max_train_epoch = args.max_train_epoch
    learning_rate_init = args.learning_rate_init
    learning_rate_min = args.learning_rate_min
    weight_decay = args.weight_decay
    use_class_weights = args.use_class_weights
    use_peak_features = args.use_peak_features
    max_channel_num = args.max_channel_num
    in_channels = args.in_channels

    label_smoothing = args.label_smoothing
    alpha = args.alpha
    scheduler_type = args.scheduler

    dsf_type = args.dsf_type
    dsf_n_out_channels = args.dsf_n_out_channels
    if dsf_n_out_channels < 0:
        dsf_n_out_channels = max_channel_num

    scheduler_type_text = "" if scheduler_type == "none" else f"-{scheduler_type}"
    label_smoothing_text = "" if label_smoothing == 0 else f"-ls{label_smoothing}"
    use_class_weights_text = "-wc" if use_class_weights else ""
    mixup_alpha_text = "" if alpha == 1 else f"-mx{alpha}"
    use_peak_features_text = "-pf" if use_peak_features else ""
    dsf_type_text = "" if dsf_type == "vanilla" else f"-{dsf_type}"
    use_simulator_text = "" if use_simulator == 0 else f"-sim{args.beat_type}"

    exp_name = f"WGAN_GP{use_simulator_text}-maxep{max_train_epoch}-c{channel_ID}-lr{learning_rate_init}-bs{batch_size}-sd{seed}"
    return exp_name


class ECG_classification_dataset_with_peak_features(Dataset):
    def __init__(self, feature_df_all_selected_p_ind_with_ecg, ecg_resampling_length=300, peak_loc_name="p_ind_resampled", label_name="label",
                 short_identifier_list=None, peak_feature_name_list=None):
        if short_identifier_list is None:
            short_identifier_list = ['patient_ID', 'interval_ID', 'block_ID', 'channel_ID', 'r_ID_abs', 'label', 'r_ID_abs_ref']
        if peak_feature_name_list is None:
            peak_feature_name_list = ["p_prom_med", "pr_int_iqr"]

        self.short_identifier_list = short_identifier_list
        self.peak_feature_name_list = peak_feature_name_list
        self.label_name = label_name
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.feature_df_all_selected_p_ind_with_ecg = feature_df_all_selected_p_ind_with_ecg
        self.ecg_resampling_length = ecg_resampling_length
        self.ecg_colnames = [f"ecg{i + 1}" for i in range(self.ecg_resampling_length)]
        self.peak_loc_name = peak_loc_name
        self.ecg_mat = self.feature_df_all_selected_p_ind_with_ecg[self.ecg_colnames].values
        self.peak_label_list = self.feature_df_all_selected_p_ind_with_ecg[self.peak_loc_name].values
        self.label_list = self.feature_df_all_selected_p_ind_with_ecg[self.label_name].values
        self.short_identifier_mat = self.feature_df_all_selected_p_ind_with_ecg[self.short_identifier_list].values
        self.peak_feature_mat = self.feature_df_all_selected_p_ind_with_ecg[self.peak_feature_name_list].values

    def __len__(self):
        return len(self.feature_df_all_selected_p_ind_with_ecg)

    def __getitem__(self, idx):
        X = self.ecg_mat[idx, :]
        peak_idx = self.peak_label_list[idx]
        label = self.label_list[idx]
        id_vec = self.short_identifier_mat[idx, :]
        peak_features = self.peak_feature_mat[idx, :]

        # return X[np.newaxis, :], peak_idx, label, id_vec, peak_features[np.newaxis, :]
        return X[np.newaxis, :], label
