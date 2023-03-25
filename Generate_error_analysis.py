import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import os
import time
from pathlib import Path
import json
import pickle

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import statsmodels.formula.api as smf

from src.methods.supervised_1d import SupervisedModel_1D
from src.models.CLOCS_1D import cnn_network_contrastive
from src.models.ResNet1D import ResNet1D

from src.utils.metrics import accuracy_at_k, weighted_mean, AUROC
from src.utils.ECG_data_loading import *

from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

from lime_timeseries import LimeTimeSeriesExplainer

torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count(), device)

softmax = torch.nn.Softmax(dim=1)


def Get_dataset_id_dict(dataset):
    dataset.return_id_vec = True
    dataset_id_dict = {}
    for i, (x, id_vec, y) in enumerate(dataset):
        ECG_signal = x.ravel()
        ECG_signal_str = "_".join([f"{ele:.3f}" for ele in ECG_signal])
        id_dict = dict(zip(dataset.short_identifier_list, id_vec))
        id_dict["label"] = y
        dataset_id_dict[ECG_signal_str] = id_dict

    return dataset_id_dict


def Lookup_ECG(ECG_signal, dataset_id_dict):
    ECG_signal = ECG_signal.ravel()
    ECG_signal_str = "_".join([f"{ele:.3f}" for ele in ECG_signal])
    id_dict = dataset_id_dict.get(ECG_signal_str)
    return id_dict


def Get_roc_curve_df_from_model_and_data_loader(model, data_loader, target_fpr=0.1, target_threshold=None, use_gpu=False):
    y_true_all_list = []
    scores_all_list = []

    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
    model = model.to(device)
    model.eval()
    auroc = AUROC(pos_label=1)

    for i, data in enumerate(data_loader):
        #         print(f"Batch {i} / {len(data_loader)}")
        X, _, y = data
        X, y = X.to(device), y.to(device)
        X_embedding = model.backbone(X)
        X_logits = model.classifier(X_embedding)
        y_true_all_list.append(y)

        scores = softmax(X_logits)[:, 1].detach().cpu()
        scores_all_list.append(scores)
        auroc.update(scores, y.detach().cpu())

    auroc_value = auroc.compute()
    auroc.reset()
    print(f"auroc_value = {auroc_value:.4f}")
    y_true_all = torch.cat(y_true_all_list, dim=0)
    scores_all = torch.cat(scores_all_list, dim=0)
    y_true_all_np = y_true_all.detach().cpu().numpy()
    y_scores = scores_all
    y_true = y_true_all_np

    auroc = auroc_value
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    roc_curve_df = pd.DataFrame([fpr, tpr, thresholds]).T
    roc_curve_df.columns = ["fpr", "tpr", "thresholds"]
    if target_threshold is not None:
        closest_threshold_idx = np.argsort(np.abs(roc_curve_df["thresholds"] - target_threshold))[0]
    else:
        closest_threshold_idx = np.argsort(np.abs(roc_curve_df["fpr"] - target_fpr))[0]
    selected_threshold = roc_curve_df["thresholds"][closest_threshold_idx]
    selected_fpr = roc_curve_df["fpr"][closest_threshold_idx]
    roc_curve_results_dict = {"roc_curve_df": roc_curve_df, "selected_threshold": selected_threshold, "selected_fpr": selected_fpr, "auroc": auroc, "y_scores": y_scores, "y_true": y_true}
    return roc_curve_results_dict


def Get_wrong_predictions_indices_from_roc_curve_results_dict(roc_curve_results_dict, target_threshold=None):
    if target_threshold is None:
        target_threshold = roc_curve_results_dict["selected_threshold"]

    y_true = roc_curve_results_dict["y_true"]
    if hasattr(roc_curve_results_dict["y_scores"], "numpy"):
        y_scores = roc_curve_results_dict["y_scores"].numpy()
    else:
        y_scores = roc_curve_results_dict["y_scores"]
    y_pred = (y_scores > target_threshold).astype(int)
    wrong_indices_dict = {0: np.where(~y_true & y_pred)[0].tolist(), 1: np.where(y_true & ~y_pred)[0].tolist()}
    return wrong_indices_dict


def Filter_data_by_id_dict(ecg_df, id_dict, num_cycle_to_show_each_side=4):
    cycle_ID = ecg_df[
        (ecg_df["patient_ID"] == id_dict["patient_ID"]) \
        & (ecg_df["interval_ID"] == id_dict["interval_ID"]) \
        & (ecg_df["block_ID"] == id_dict["block_ID"]) \
        & (ecg_df["channel_ID"] == id_dict["channel_ID"])
        & (ecg_df["r_ID_abs"] == id_dict["r_ID_abs"])
        ]["cycle_ID"].values[0]
    ecg_df_selected = ecg_df[
        (ecg_df["patient_ID"] == id_dict["patient_ID"]) \
        & (ecg_df["interval_ID"] == id_dict["interval_ID"]) \
        & (ecg_df["block_ID"] == id_dict["block_ID"]) \
        & (ecg_df["channel_ID"] == id_dict["channel_ID"]) \
        & ((ecg_df["cycle_ID"] <= cycle_ID + num_cycle_to_show_each_side) \
           & (ecg_df["cycle_ID"] >= cycle_ID - num_cycle_to_show_each_side))
        ].sort_values(by=["cycle_ID"], ascending=True)
    return ecg_df_selected, cycle_ID


def Show_LIME_explanation_for_idx_with_id(idx_list, dataset, NN_predict_proba, dataset_id_dict, dataset_df,
                                          class_names=None, num_slices=30, num_cycle_to_show_each_side=4,
                                          num_features=10, target_threshold=0.5,
                                          replacement_method="total_mean", entropy_list=None):
    if class_names is None:
        class_names = ['Sinus', 'JET']

    nrow = len(idx_list)
    ncol = 2 * num_cycle_to_show_each_side + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4 * nrow))
    for i, idx in enumerate(idx_list):
        ecg_signal, _, label = dataset[idx]
        id_dict = Lookup_ECG(ecg_signal, dataset_id_dict)
        ecg_df_selected, cycle_ID = Filter_data_by_id_dict(dataset_df, id_dict,
                                                           num_cycle_to_show_each_side=num_cycle_to_show_each_side)
        #         print(ecg_df_selected)
        #         assert len(ecg_df_selected) == ncol
        for j in range(len(ecg_df_selected)):
            if nrow == 1:
                ax = axes[j]
            elif ncol == 1:
                ax = axes[i]
            else:
                ax = axes[i, j]

            cycle_ID_j = ecg_df_selected["cycle_ID"].values[j]
            patient_ID = ecg_df_selected["patient_ID"].values[j]
            ecg_signal = Normalize(ecg_df_selected[dataset.ecg_colnames].values[j, :])

            if j == len(ecg_df_selected) // 2:
                entropy_text = f", Ent = {entropy_list[i]:.3f}" if entropy_list is not None else ""
                ax.set_title(f"Patient ID: {patient_ID}{entropy_text}")

            ax.set_ylabel(f"Cycle {cycle_ID_j}")

            explainer = LimeTimeSeriesExplainer(class_names=class_names)
            exp = explainer.explain_instance(
                ecg_signal, NN_predict_proba, num_features=num_features,
                num_samples=5000, num_slices=num_slices,
                replacement_method=replacement_method)

            Plot_LIME_explanation(ecg_signal, label, NN_predict_proba, exp, target_threshold=target_threshold,
                                  num_slices=num_slices, num_features=num_features, ax=ax)


#             return id_dict

def Plot_LIME_explanation(ecg_signal, label, NN_predict_proba, exp,
                          num_slices=30, num_features=10, ax=None, target_threshold=0.5):
    if ax is None:
        ax = plt.gca()

    sns.set(style="white", font_scale=1.5)
    proba = NN_predict_proba(ecg_signal[np.newaxis, :]).ravel()
    values_per_slice = np.ceil(len(ecg_signal) / num_slices).astype(int)
    label_dict = {0: "Sinus", 1: "JET"}
    color_dict = {0: cm.tab10(0), 1: cm.tab10(1)}
    for i, p in enumerate(proba):
        ax.text(20, 0.9 - 0.1 * i,
                f"Pr(y={label_dict[i]}|x; CNN) = {p:.2f}", size=15)

    if proba[1] < target_threshold and label == 1:
        error = "FN"
        error_color = f"#FF00FF"
    elif proba[1] > target_threshold and label == 0:
        error = "FP"
        error_color = f"#FF0000"
    else:
        error = None
        error_color = None

    ax.plot(ecg_signal, color=color_dict[label], label=f"True label: {label_dict[label]}")
    if error is not None:
        ax.text(0.05 * len(ecg_signal), 0.1, error, color=error_color)
    ax.legend(loc='lower center')

    slice_intervals = np.arange(0, len(ecg_signal) + 1, values_per_slice)
    slice_interval_midpoint_list = [(slice_intervals[i] + slice_intervals[i + 1]) / 2 \
                                    for i in range(len(slice_intervals) - 1)]
    interval_val_tuple_list = []
    for i in range(num_features):
        feature, weight = exp.as_list()[i]
        start = feature * values_per_slice
        end = start + values_per_slice
        color = 'red' if weight < 0 else 'green'
        ax.axvspan(start, end, color=color, alpha=abs(weight * 2))
        #         print(i, start, end)
        #         ax.text(start, 0.1, f"{weight:.2f}") # This is the coefficient of the feature (slice)
        # learned by the underlying linear model
        interval_val_tuple_list.append((start, end, weight))
    ax_twinx = ax.twinx()

    interval_val_list = []
    for slice_interval_midpoint in slice_interval_midpoint_list:
        for j, interval_val_tuple in enumerate(interval_val_tuple_list):
            if slice_interval_midpoint > interval_val_tuple[0] and slice_interval_midpoint < interval_val_tuple[1]:
                interval_val_list.append(interval_val_tuple[2])
                break
        else:
            interval_val_list.append(0)

    #     print(len(slice_interval_midpoint_list), slice_interval_midpoint_list)
    #     print(len(interval_val_list), interval_val_list)
    ax_twinx.bar(x=slice_interval_midpoint_list, height=interval_val_list, color="#000000", width=0.5 * values_per_slice)
    ax_twinx.set_ylim(np.minimum(0, 2 * np.min(interval_val_list)),
                      np.maximum(0, 2 * np.max(interval_val_list)))
    ax_twinx.set_ylim(-0.25, 0.25)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax_twinx.set_yticklabels([])


def Get_NN_predict_proba(model):
    model.eval()

    def NN_predict_proba(X):
        X = torch.from_numpy(X)
        X = X.unsqueeze(1)
        X_embedding = model.backbone(X.cpu()).cpu()
        X_logits = model.classifier(X_embedding.cpu()).cpu()
        predict_proba = softmax(X_logits).detach().cpu().numpy()
        return predict_proba

    return NN_predict_proba


data_folder = "/mnt/scratch07/yilong"
data_chunk_folder = "ecg-pat40-tch-sinus_jet_lead2"
load_training_data = True

patient_ID_list_train = [398573, 462229, 637891, 667681, 537854, 628521, 642321, 662493,
                         387479, 624179, 417349, 551554, 631270, 655769, 678877]  # 15
patient_ID_list_test = [756172, 424072, 748555, 748900, 759678, 741235, 595561, 678607,
                        782501, 510915, 771495, 740475, 533362, 581650, 803389, 577874,
                        681150, 536886, 477589, 844864, 824744, 515544, 771958, 725860, 609090]  # 25
patient_ID_list_val = [462229, 642321, 387479]  # 3
patient_ID_list_dev = [patient_ID for patient_ID in patient_ID_list_train if
                       patient_ID not in patient_ID_list_val]  # 12

if load_training_data:
    data_chunk_list = []
    for data_filename in os.listdir(os.path.join(data_folder, data_chunk_folder)):
        data_chunk_list.append(pd.read_csv(os.path.join(data_folder, data_chunk_folder, data_filename)))
    feature_df_all_selected_with_ecg = pd.concat(data_chunk_list, axis=0)
    channel_ID = 2
    feature_with_ecg_df_dev_single_lead = feature_df_all_selected_with_ecg.query(f"patient_ID in {patient_ID_list_dev}").query(f"channel_ID == {channel_ID}")
    print(f"[Dataset Loaded] dev data: {feature_with_ecg_df_dev_single_lead.shape}")

    feature_with_ecg_df_val_single_lead = feature_df_all_selected_with_ecg.query(f"patient_ID in {patient_ID_list_val}").query(f"channel_ID == {channel_ID}")
    print(f"[Dataset Loaded] val data: {feature_with_ecg_df_val_single_lead.shape}")
    feature_with_ecg_df_test_single_lead = feature_df_all_selected_with_ecg.query(f"patient_ID in {patient_ID_list_test}").query(f"channel_ID == {channel_ID}")
    print(f"[Dataset Loaded] test data: {feature_with_ecg_df_test_single_lead.shape}")
else:
    feature_with_ecg_df_dev_single_lead = pd.DataFrame()
    dev_dataset = None

# feature_with_ecg_df_val_single_lead = pd.read_csv("feature_with_ecg_df_val_lead2.csv")
# feature_with_ecg_df_test_single_lead = pd.read_csv("feature_with_ecg_df_test_lead2.csv")
# print(f"[Dataset Loaded] val data: {feature_with_ecg_df_val_single_lead.shape}")
# print(f"[Dataset Loaded] test data: {feature_with_ecg_df_test_single_lead.shape}")

if load_training_data:
    dev_dataset = ECG_classification_dataset_with_peak_features(feature_with_ecg_df_dev_single_lead,
                                                                shift_signal=False,
                                                                shift_amount=0,
                                                                normalize_signal=True,
                                                                ecg_resampling_length_target=300,
                                                                return_id_vec=True)
    print(f"[Dataset Created] dev data: {len(dev_dataset)}")

val_dataset = ECG_classification_dataset_with_peak_features(feature_with_ecg_df_val_single_lead,
                                                            shift_signal=False,
                                                            shift_amount=0,
                                                            normalize_signal=True,
                                                            ecg_resampling_length_target=300,
                                                            return_id_vec=True)
print(f"[Dataset Created] val data: {len(val_dataset)}")

test_dataset = ECG_classification_dataset_with_peak_features(feature_with_ecg_df_test_single_lead,
                                                             shift_signal=False,
                                                             shift_amount=0,
                                                             normalize_signal=True,
                                                             ecg_resampling_length_target=300,
                                                             return_id_vec=True)
print(f"[Dataset Created] test data: {len(test_dataset)}")

if load_training_data:
    dataset_id_dict_dev = None
#     dataset_id_dict_dev = Get_dataset_id_dict(dev_dataset)
#     print(f"[Dataset ID Dict Created] dev data: {len(dataset_id_dict_dev)}")
else:
    dataset_id_dict_dev = None

dataset_id_dict_val = Get_dataset_id_dict(val_dataset)
print(f"[Dataset ID Dict Created] val data: {len(dataset_id_dict_val)}")
dataset_id_dict_test = Get_dataset_id_dict(test_dataset)
print(f"[Dataset ID Dict Created] test data: {len(dataset_id_dict_test)}")

#%% Load models
model_dict = {}
model_ID_dict = {}
model_ID = "s6qzuo6i"
ckpt_dir = Path(f"trained_models/linear/{model_ID}") # test_auroc = 0.9631
max_val_auroc = max([float(ele.split("=")[-1].replace(".ckpt", "")) \
                     for ele in os.listdir(ckpt_dir) if "=" in ele and "val_auroc" in ele])
ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(f"val_auroc={max_val_auroc}.ckpt")][0]
args_path = ckpt_dir / "args.json"
print(f"ckpt_path: {ckpt_path}")
with open(args_path) as f:
    method_args_default = json.load(f)
method_args_default

#%% ViT-1D
from src.models.ViT_1D import ViT

model_folder_name = "supervised-ViT_1D-ECG_normalized-20230220_v35-maxep200-es_patience40-mixup-ls-aug_selected_20221029-prob1.25_2-seed0-best"
model_folder_path = Path(f"trained_models/linear/{model_folder_name}")
config_filename = [ele for ele in os.listdir(model_folder_path) if ".csv" in ele][0]
config_df = pd.read_csv(f"trained_models/linear/{model_folder_name}/{config_filename}")
model_ID_list = [ele for ele in os.listdir(os.path.join(model_folder_path)) if ".csv" not in ele]
for i, model_ID in enumerate(model_ID_list):
    #     model_ID = "s6qzuo6i"
    print(f"model_ID = {model_ID}")
    config_df_i = config_df.query(f"ID == '{model_ID}'")

    model_key = f"ViT1D_ens_{i}"
    model_ID_dict[model_key] = model_ID
    embedding_dim = 256
    model = ViT(
        seq_len=300,
        patch_size=15,
        num_classes=2,
        dim=256,
        depth=config_df_i["num_layers"].values[0],
        heads=config_df_i["nhead"].values[0],
        mlp_dim=config_df_i["dim_feedforward"].values[0],
        dropout=0.1,
        emb_dropout=0.1,
        channels=1, dim_head=config_df_i["d_model"].values[0]
    )
    model.fc = nn.Identity()
    ckpt_dir = Path(f"trained_models/linear/{model_folder_name}/{model_ID}")  # test_auroc = 0.9631
    max_val_auroc = max([float(ele.split("=")[-1].replace(".ckpt", "")) \
                         for ele in os.listdir(ckpt_dir) if "=" in ele and "val_auroc" in ele])
    ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(f"val_auroc={max_val_auroc:.4f}.ckpt")][0]
    method_args = method_args_default
    model.pretrained_occlusion_model_dict = None
    method_args["backbone"] = model
    model_loaded = SupervisedModel_1D.load_from_checkpoint(
        ckpt_path, strict=False, **method_args
    )
    model_dict[model_key] = model_loaded

#%% Get softmax scores from models
batch_size = 32
num_workers = 0
pin_memory = False
if load_training_data:
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=pin_memory
    )
    print(f"[Dataloader Created] dev data: {len(dev_loader)}")
else:
    dev_loader = None

val_loader = DataLoader(
    val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=pin_memory
)
print(f"[Dataloader Created] val data: {len(val_loader)}")
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=pin_memory
)
print(f"[Dataloader Created] test data: {len(test_loader)}")

df_dict = {
    "dev": feature_with_ecg_df_dev_single_lead,
    "val": feature_with_ecg_df_val_single_lead,
    "test": feature_with_ecg_df_test_single_lead
}
dataset_dict = {
    "dev": dev_dataset,
    "val": val_dataset,
    "test": test_dataset
}
dataset_id_dict_dict = {
    "dev": dataset_id_dict_dev,
    "val": dataset_id_dict_val,
    "test": dataset_id_dict_test
}
data_loader_dict = {
    "dev": dev_loader,
    "val": val_loader,
    "test": test_loader
}
roc_curve_results_dict_dict_dict = {"dev": {}, "val": {}, "test": {}}

torch.set_num_threads(4)
torch.get_num_threads()

roc_curve_results_dict_dict_dict_folder_name = f"roc_curve_results_dict_dict_dict_folder"
os.makedirs(f"roc_curve_results_dict_dict_dict_folder", exist_ok=True)

target_fpr = 0.05
st = time.time()
for mode in ["val", "test"]:
    for model_key in model_dict:
        use_gpu = True
#         use_gpu = False
        model_ID = model_ID_dict[model_key]
        save_name = f"roc_curve_results_dict_dict_dict-{model_ID}-{mode}.pickle"
        save_path = os.path.join(roc_curve_results_dict_dict_dict_folder_name, save_name)
        print(f"[Time {time.time() - st:.1f}]", mode, model_key, f"use_gpu = {use_gpu}")
        if model_key not in roc_curve_results_dict_dict_dict[mode]:
            if os.path.exists(save_path):
                with open(save_path, "rb") as f:
                    roc_curve_results_dict_dict_tmp = pickle.load(f)
                roc_curve_results_dict_dict_dict[mode][model_key] = roc_curve_results_dict_dict_tmp
                print(f"Loaded.")
            else:
                roc_curve_results_dict_dict_dict[mode][model_key] = Get_roc_curve_df_from_model_and_data_loader(
                    model_dict[model_key], data_loader_dict[mode], target_fpr=target_fpr, target_threshold=None)
                with open(save_path, "wb") as f:
                    pickle.dump(roc_curve_results_dict_dict_dict[mode][model_key], f)
                print(f"Saved.")
#             torch.cuda.empty_cache()
        else:
            print(f"Skipped.")

