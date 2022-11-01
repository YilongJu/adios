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
from scipy.signal import resample_poly
from src.transforms.ecg_transform_1d import *

patient_ID_list_train = [398573, 462229, 637891, 667681, 537854, 628521, 642321, 662493,
                         387479, 624179, 417349, 551554, 631270, 655769, 678877]  # 15
patient_ID_list_test = [756172, 424072, 748555, 748900, 759678, 741235, 595561, 678607,
                        782501, 510915, 771495, 740475, 533362, 581650, 803389, 577874,
                        681150, 536886, 477589, 844864, 824744, 515544, 771958, 725860, 609090]  # 25
patient_ID_list_val = [462229, 642321, 387479] # 3
patient_ID_list_dev = [patient_ID for patient_ID in patient_ID_list_train if patient_ID not in patient_ID_list_val] # 12


def Data_preprocessing(args):
    """ Preprocessing """
    if args.cluster_name in ["b1", "b3", "b4"]:
        data_folder = os.path.normpath("/mnt/scratch07/yilong")
    elif args.cluster_name in ["b2"]:
        data_folder = os.path.normpath("/mnt/group1/yilong/JET-Detection-Data")
    else:
        if os.path.exists("/mnt/scratch07/yilong"):
            data_folder = "/mnt/scratch07/yilong"
        elif os.path.exists("/mnt/group1/yilong/JET-Detection-Data"):
            data_folder = "/mnt/group1/yilong/JET-Detection-Data"
        else:
            raise ValueError("Cannot determine data_folder!")

    # save_folder = os.path.join(data_folder, "Results")
    save_folder = data_folder

    if platform.system() == "Darwin":
        print("Using MacOS.")
        data_folder = os.path.normpath("/Users/yj31/Dropbox/Study/GitHub/JET-Detection")
    elif platform.system() == "Linux":
        print("Using Linux.")
    else:
        print("Using Windows.")
        data_folder = os.path.normpath("D:\\Dropbox\\Study\\GitHub\\JET-Detection")
        # data_folder = os.path.normpath("D:\\Backup\\JET-Detection\\")

    if args.dataset in ["ecg-TCH-40_patient-20220201"]:
        # debug = True
        # debug = False
        debug = args.debug
        if debug:
            feature_df_all_selected_with_ecg = pd.read_csv(
                os.path.join(data_folder, "feature_df_all_selected_with_ecg_20220210_rtfixed_sample10000.csv"))
        else:
            if args.read_data_by_chunk:
                if args.channel_ID == 2:
                    data_chunk_folder = "ecg-pat40-tch-sinus_jet_lead2"
                else:
                    data_chunk_folder = args.data_chunk_folder

                data_chunk_list = []
                for data_filename in os.listdir(os.path.join(data_folder, data_chunk_folder)):
                    data_chunk_list.append(pd.read_csv(os.path.join(data_folder, data_chunk_folder, data_filename)))
                feature_df_all_selected_with_ecg = pd.concat(data_chunk_list, axis=0)
            else:
                feature_df_all_selected_with_ecg = pd.read_csv(
                    os.path.join(data_folder, "feature_df_all_selected_with_ecg_20220210_rtfixed.csv"))
        feature_with_ecg_df_train = feature_df_all_selected_with_ecg.query(f"patient_ID in {patient_ID_list_train}")
        feature_with_ecg_df_test = feature_df_all_selected_with_ecg.query(f"patient_ID in {patient_ID_list_test}")
        feature_with_ecg_df_dev = feature_df_all_selected_with_ecg.query(f"patient_ID in {patient_ID_list_dev}")
        feature_with_ecg_df_val = feature_df_all_selected_with_ecg.query(f"patient_ID in {patient_ID_list_val}")
        print(f"Data shape: {feature_df_all_selected_with_ecg.shape}, train: {feature_with_ecg_df_train.shape}, dev: {feature_with_ecg_df_dev.shape}, val: {feature_with_ecg_df_val.shape}, test: {feature_with_ecg_df_test.shape}")
    elif args.dataset in ["ecg-TCH-40_patient-20220201_with_CVP"]:
        feature_with_ecg_df_train = None
        debug = args.debug
        if debug:
            feature_with_ecg_df_dev = np.load(os.path.join(data_folder, "ECG_CVP_20221101_dev_10000.npz"))
        else:
            feature_with_ecg_df_dev = np.load(os.path.join(data_folder, "ECG_CVP_20221101_dev.npz"))
        feature_with_ecg_df_val = np.load(os.path.join(data_folder, "ECG_CVP_20221101_val.npz"))
        feature_with_ecg_df_test = np.load(os.path.join(data_folder, "ECG_CVP_20221101_test.npz"))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return feature_with_ecg_df_train, feature_with_ecg_df_test, feature_with_ecg_df_dev, feature_with_ecg_df_val, save_folder
    # return feature_with_ecg_df_train, feature_with_ecg_df_test, save_folder


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

def Normalize(vec, eps=1e-8):
    """ Normalize a 1d vector to 0-1 range """
    vec = vec - np.min(vec)
    vec = vec / np.max(vec + eps)
    return vec

def Lower(word):
    """ Convert word to lower case """
    return word.lower()

def Transform_frame(frame, transforms, aug_prob=0.0, dataset_name="ecg-TCH-40_patient-20220201"):
    # Adapted from https://github.com/danikiyasseh/CLOCS

    """ Apply Sequence of Perturbations to Frame
    Args:
        frame (numpy array): frame containing ECG data
    Outputs
        frame (numpy array): perturbed frame based
    """
    if Lower('Gaussian') in transforms:
        frame = Add_Gaussian_noise(frame, dataset_name=dataset_name)

    if Lower('FlipAlongY') in transforms:
        frame = Flip_Along_Y(frame)

    if Lower('FlipAlongX') in transforms:
        frame = Flip_Along_X(frame)

    if Lower("Transverse") in transforms:
        frame = Transverse_transformation(frame)

    if Lower("Longitudinal") in transforms:
        frame = Longitudinal_transformation(frame)

    if Lower("TemporalWarp") in transforms: # The TaskAug Paper (using init values)
        frame = Temporal_Warp(frame)

    if Lower("BaselineWander") in transforms: # The TaskAug Paper (using init values)
        frame = Baseline_wander(frame)

    if Lower("GauNoise") in transforms: # The TaskAug Paper (using init values)
        # No point of using this since we have already added Gaussian noise
        frame = Gau_noise(frame)
        # ------------------------------------!

    if Lower("MagnitudeScale") in transforms: # The TaskAug Paper (using init values)
        # No point of using this since we normalize the signal to 0-1 range
        frame = Magnitude_scale(frame)
        # ------------------------------------!

    if Lower("TimeMask") in transforms: # The TaskAug Paper (using init values)
        frame = Time_mask(frame)

    if Lower("RandTemporalDisp") in transforms: # The TaskAug Paper (using init values)
        frame = Random_temporal_displacement(frame)

    if Lower("SpecAugment") in transforms: # The TaskAug Paper (baseline)
        """ Does not make sense to mask by freq since we only have 1 heartbeat """
        """ Masking by time is the same as TimeMask """
        pass

    if Lower("DiscGuidedWarp") in transforms: # The TaskAug Paper (baseline), discriminative guided warping (DGW)
        """ Slow and there are artifacts """
        pass

    if Lower("SMOTE") in transforms: # The TaskAug Paper (baseline) upsampling for classes that have less samples
        """ Not really useful since we have a lot of samples """
        pass

    if Lower("SelectedAug_20221025") in transforms:
        """ Selected Augmentations based on test performance """
        """ Using Longitudinal (better than TemporalWarp), Transverse (better than BaselineWander),
            RandTemporalDisp, Gaussian, FlipAlongX """
        if np.random.uniform() < aug_prob:
            frame = Longitudinal_transformation(frame)
        if np.random.uniform() < aug_prob:
            frame = Transverse_transformation(frame)
        if np.random.uniform() < aug_prob:
            frame = Random_temporal_displacement(frame)
        if np.random.uniform() < aug_prob:
            frame = Add_Gaussian_noise(frame, dataset_name=dataset_name)
        if np.random.uniform() < aug_prob:
            frame = Flip_Along_X(frame)

    if Lower("SelectedAug_20221029") in transforms:
        """ Selected Augmentations based on test performance """
        """ Using Longitudinal (better than TemporalWarp), Transverse (better than BaselineWander),
            RandTemporalDisp, Gaussian """
        """ Only apply one augmentation """
        def Add_Gaussian_noise_dataset(x):
            return Add_Gaussian_noise(x, dataset_name=dataset_name)

        transformation_func_list = np.random.choice(
            [Longitudinal_transformation, Transverse_transformation, Random_temporal_displacement,
             Add_Gaussian_noise_dataset], size=2, replace=False)
        random_number = np.random.uniform()

        if random_number < aug_prob:
            frame = transformation_func_list[0](frame)

        if random_number + 1 < aug_prob:
            frame = transformation_func_list[1](frame)

    # Keep data in 0-1 range
    frame = Normalize(frame)
    return frame


class ECG_classification_dataset_with_peak_features(Dataset):
    def __init__(self, feature_df_all_selected_p_ind_with_ecg, ecg_resampling_length_target=300,
                 peak_loc_name="p_ind_resampled", label_name="label", short_identifier_list=None,
                 peak_feature_name_list=None, shift_signal=False, shift_amount=None, normalize_signal=False,
                 transforms=None, dataset_name="ecg-TCH-40_patient-20220201", aug_prob=0):
        """
        normalize_signal: Normalize each individual signal to 0 - 1 range
        """
        # print(f"ecg_resampling_length_target: {ecg_resampling_length_target}")
        if short_identifier_list is None:
            short_identifier_list = ['patient_ID', 'interval_ID', 'block_ID', 'channel_ID', 'r_ID_abs', 'label',
                                     'r_ID_abs_ref']
        if peak_feature_name_list is None:
            peak_feature_name_list = ["p_prom_med", "pr_int_iqr"]

        if transforms is None:
            self.transforms = []
        else:
            if isinstance(transforms, str):
                transforms = [transforms]
            self.transforms = [Lower(ele) for ele in transforms]

        self.dataset_name = dataset_name
        self.short_identifier_list = short_identifier_list
        self.peak_feature_name_list = peak_feature_name_list
        self.label_name = label_name
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.feature_df_all_selected_p_ind_with_ecg = feature_df_all_selected_p_ind_with_ecg
        self.ecg_resampling_length = 300
        self.ecg_resampling_length_target = ecg_resampling_length_target
        self.ecg_colnames = [f"ecg{i + 1}" for i in range(self.ecg_resampling_length)]
        self.peak_loc_name = peak_loc_name
        self.ecg_mat = self.feature_df_all_selected_p_ind_with_ecg[self.ecg_colnames].values
        self.peak_label_list = self.feature_df_all_selected_p_ind_with_ecg[self.peak_loc_name].values
        self.label_list = self.feature_df_all_selected_p_ind_with_ecg[self.label_name].values
        self.short_identifier_mat = self.feature_df_all_selected_p_ind_with_ecg[self.short_identifier_list].values
        self.peak_feature_mat = self.feature_df_all_selected_p_ind_with_ecg[self.peak_feature_name_list].values

        self.shift_signal = shift_signal
        self.shift_amount = shift_amount
        self.normalize_signal = normalize_signal
        self.aug_prob = aug_prob
        if self.shift_signal:
            if self.shift_amount is None:
                self.shift_amount = 0
            self.ecg_mat -= self.shift_amount  # Shift ECG to 0 baseline

        if self.normalize_signal:
            ecg_min = np.min(self.ecg_mat, axis=1)[:, np.newaxis]
            ecg_max = np.max(self.ecg_mat, axis=1)[:, np.newaxis]
            self.ecg_mat = (self.ecg_mat - ecg_min) / (ecg_max - ecg_min)

    def obtain_perturbed_frame(self, frame):
        frame = Transform_frame(frame, transforms=self.transforms, dataset_name=self.dataset_name, aug_prob=self.aug_prob)
        return frame

    def __len__(self):
        return len(self.feature_df_all_selected_p_ind_with_ecg)

    def __getitem__(self, idx):
        X = self.ecg_mat[idx, :]
        if self.ecg_resampling_length_target != self.ecg_resampling_length:
            X = resample_poly(X, int(self.ecg_resampling_length_target / 100), int(self.ecg_resampling_length / 100),
                              padtype="line")
        X = Normalize(X)
        X_aug = self.obtain_perturbed_frame(X)
        peak_idx = self.peak_label_list[idx]
        label = self.label_list[idx]
        id_vec = self.short_identifier_mat[idx, :]
        peak_features = self.peak_feature_mat[idx, :]

        # return X[np.newaxis, :], peak_idx, label, id_vec, peak_features[np.newaxis, :]
        if len(self.transforms) == 0 \
                or (len(self.transforms) == 1 and Lower(self.transforms[0]) in [Lower("Identity"), Lower("SelectedAug_20221025"), Lower("SelectedAug_20221029")]):
            return X_aug[np.newaxis, :], label
        else:
            return (X[np.newaxis, :], X_aug[np.newaxis, :]), label


class ECG_classification_dataset_with_CVP(Dataset):
    def __init__(self, data_tensor_np, data_label_np, in_channels=1, ecg_resampling_length=300, shift_signal=False, shift_amount=None, normalize_signal=False,
                 transforms=None, dataset_name="ecg-TCH-40_patient-20220201_with_CVP", aug_prob=0, ecg_resampling_length_target=300):
        self.data_tensor_np = data_tensor_np
        self.data_label_np = data_label_np
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.in_channels = in_channels
        self.ecg_resampling_length = ecg_resampling_length
        self.ecg_resampling_length_target = ecg_resampling_length_target

        if transforms is None:
            self.transforms = []
        else:
            if isinstance(transforms, str):
                transforms = [transforms]
            self.transforms = [Lower(ele) for ele in transforms]
        self.dataset_name = dataset_name

        self.shift_signal = shift_signal
        self.shift_amount = shift_amount
        self.normalize_signal = normalize_signal
        self.aug_prob = aug_prob
        if self.shift_signal:
            if self.shift_amount is None:
                self.shift_amount = 0
            self.data_tensor_np -= self.shift_amount  # Shift ECG to 0 baseline

        if self.normalize_signal:
            ecg_min = np.min(self.data_tensor_np, axis=2)[:, :, np.newaxis]
            ecg_max = np.max(self.data_tensor_np, axis=2)[:, :, np.newaxis]
            self.data_tensor_np = (self.data_tensor_np - ecg_min) / (ecg_max - ecg_min)

    def __len__(self):
        return len(self.data_label_np)

    def obtain_perturbed_frame(self, frame):
        frame_list = []
        for x in frame:
            frame_list.append(Transform_frame(x, transforms=self.transforms, dataset_name=self.dataset_name, aug_prob=self.aug_prob)[np.newaxis, :])
        frame = np.concatenate(frame_list, axis=0)
        # print(f"augmented frame shape: {frame.shape}")
        return frame

    def __getitem__(self, idx):
        X = self.data_tensor_np[idx, :self.in_channels, ...]
        X_aug = self.obtain_perturbed_frame(X)

        label = self.data_label_np[idx]
        if len(self.transforms) == 0 \
                or (len(self.transforms) == 1 and Lower(self.transforms[0]) in [Lower("Identity"),
                                                                                Lower("SelectedAug_20221025"),
                                                                                Lower("SelectedAug_20221029")]):
            return X_aug, label
        else:
            return (X, X_aug), label
