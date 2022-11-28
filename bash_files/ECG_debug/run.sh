# SimCLR
python main_pretrain.py --seed 0 --alpha_entropy 0 --alpha_sparsity 1. --batch_size 256 --classifier_lr 0.1 --dataset "ecg-TCH-40_patient-20220201" --lr 0.13234838295784523 --mask_lr 0.01946841419435407 --optimizer sgd --scheduler warmup_cosine --weight_decay 8.504658547335148e-05 --temperature 0.2 --max_epochs 1 --N 1 --n_crops 1 --n_small_crops 0 --encoder resnet1d --mask_fbase 128 --method simclr_adios_1d --output_dim 256 --proj_hidden_dim 2048 --unet_norm gn --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --wandb_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios" --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models" --project "adios_ecg_debug" --entity "yilongju" --name "simclr_adios_resnet18_ECG_debug" --wandb True --ptl_accelerator "cpu" --debug
#    --gpus 0
# [20221128] Adding data augmentation and using clocs1d
python main_pretrain.py --ecg_resampling_length_target 300 --stride 2 --c4_multiplier 3 --transforms "SelectedAug_20221029" --aug_prob 1.5 --seed 0 --alpha_entropy 0 --alpha_sparsity 1. --batch_size 256 --classifier_lr 0.1 --dataset "ecg-TCH-40_patient-20220201" --lr 0.13234838295784523 --mask_lr 0.01946841419435407 --optimizer sgd --scheduler warmup_cosine --weight_decay 8.504658547335148e-05 --temperature 0.2 --max_epochs 2 --N 1 --n_crops 1 --n_small_crops 0 --encoder "clocs_cnn1d" --mask_fbase 16 --method simclr_adios_1d --output_dim 128 --proj_hidden_dim 256 --unet_norm gn --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --wandb_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios" --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models" --project "adios_ecg_debug" --entity "yilongju" --name "simclr_adios_resnet18_ECG_debug" --wandb True --ptl_accelerator "cpu" --debug


# BYOL
python main_pretrain.py --seed 0 --alpha_entropy 0 --alpha_sparsity 1. --batch_size 256 --classifier_lr 0.1 --dataset "ecg-TCH-40_patient-20220201" --lr 0.13234838295784523 --mask_lr 0.01946841419435407 --optimizer sgd --scheduler warmup_cosine --weight_decay 8.504658547335148e-05 --max_epochs 1 --N 2 --n_crops 1 --n_small_crops 0 --encoder resnet1d --mask_fbase 128 --method byol_adios_1d --output_dim 256 --proj_hidden_dim 2048 --unet_norm gn --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --wandb_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios" --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models" --project "byol-adios_ecg_debug" --entity "yilongju" --name "byol_adios_resnet18_ECG_debug" --wandb True --ptl_accelerator "cpu" --debug

# SimSiam
python main_pretrain.py --seed 0 --alpha_entropy 0 --alpha_sparsity 1. --batch_size 256 --classifier_lr 0.1 --dataset "ecg-TCH-40_patient-20220201" --lr 0.13234838295784523 --mask_lr 0.01946841419435407 --optimizer sgd --scheduler warmup_cosine --weight_decay 8.504658547335148e-05 --max_epochs 1 --N 2 --n_crops 1 --n_small_crops 0 --encoder resnet1d --mask_fbase 128 --method simsiam_adios_1d --output_dim 256 --proj_hidden_dim 2048 --unet_norm gn --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --wandb_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios" --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models" --project "byol-adios_ecg_debug" --entity "yilongju" --name "byol_adios_resnet18_ECG_debug" --wandb True --ptl_accelerator "cpu" --debug

python main_pretrain.py --seed 0 --alpha_entropy 0 --alpha_sparsity 1. --batch_size 256 --classifier_lr 0.1 --dataset "ecg-TCH-40_patient-20220201" --lr 0.13234838295784523 --mask_lr 0.01946841419435407 --optimizer sgd --scheduler warmup_cosine --weight_decay 8.504658547335148e-05 --max_epochs 4 --N 1 --n_crops 1 --n_small_crops 0 --encoder resnet1d --mask_fbase 128 --method byol_adios_1d --output_dim 256 --proj_hidden_dim 2048 --unet_norm gn --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --wandb_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios" --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models" --project "byol-adios_ecg_debug" --entity "yilongju" --name "byol_adios_resnet18_ECG_debug" --wandb True --ptl_accelerator "cpu"



CUDA_VISIBLE_DEVICES=0 python main_pretrain.py --seed 0 --alpha_entropy 0 --alpha_sparsity 1. --batch_size 256 --classifier_lr 0.1 --dataset "ecg-TCH-40_patient-20220201" --lr 0.13234838295784523 --mask_lr 0.01946841419435407 --optimizer sgd --scheduler warmup_cosine --weight_decay 8.504658547335148e-05 --temperature 0.2 --max_epochs 400 --N 6 --n_crops 1 --n_small_crops 0 --encoder resnet1d --mask_fbase 128 --method simclr_adios_1d --output_dim 256 --proj_hidden_dim 2048 --unet_norm gn --data_dir "/mnt/group1/yilong/JET-Detection-Data" --wandb_dir "/home/yilong/Github/adios" --checkpoint_dir "/home/yilong/Github/adios/trained_models" --project "adios_ecg_cluster_debug" --entity "yilongju" --name "simclr_adios_resnet18_ECG_cluster_debug" --wandb True --ptl_accelerator "ddp" --gpus 0 --cluster_name "b2"

CUDA_VISIBLE_DEVICES=1 python main_pretrain.py --seed 0 --alpha_entropy 0 --alpha_sparsity 1. --batch_size 256 --classifier_lr 0.1 --dataset "ecg-TCH-40_patient-20220201" --lr 0.13234838295784523 --mask_lr 0.01946841419435407 --optimizer sgd --scheduler warmup_cosine --weight_decay 8.504658547335148e-05 --temperature 0.2 --max_epochs 400 --N 2 --n_crops 1 --n_small_crops 0 --encoder resnet1d --mask_fbase 128 --method simclr_adios_1d --output_dim 256 --proj_hidden_dim 2048 --unet_norm gn --data_dir "/mnt/group1/yilong/JET-Detection-Data" --wandb_dir "/home/yilong/Github/adios" --checkpoint_dir "/home/yilong/Github/adios/trained_models" --project "adios_ecg_cluster_debug_Nmask2" --entity "yilongju" --name "simclr_adios_resnet18_ECG_cluster_debug_Nmask2" --wandb True --ptl_accelerator "ddp" --gpus 0 --cluster_name "b2"


CUDA_VISIBLE_DEVICES=2 python main_pretrain.py --seed 0 --alpha_entropy 0 --alpha_sparsity 1. --batch_size 256 --classifier_lr 0.1 --dataset "ecg-TCH-40_patient-20220201" --lr 0.13234838295784523 --mask_lr 0.01946841419435407 --optimizer sgd --scheduler warmup_cosine --weight_decay 8.504658547335148e-04 --temperature 0.2 --max_epochs 400 --N 6 --n_crops 1 --n_small_crops 0 --encoder resnet1d --mask_fbase 128 --method simclr_adios_1d --output_dim 256 --proj_hidden_dim 2048 --unet_norm gn --data_dir "/mnt/group1/yilong/JET-Detection-Data" --wandb_dir "/home/yilong/Github/adios" --checkpoint_dir "/home/yilong/Github/adios/trained_models" --project "adios_ecg_cluster_debug" --entity "yilongju" --name "simclr_adios_resnet18_ECG_cluster_debug_10x_more_decay" --wandb True --ptl_accelerator "ddp" --gpus 0 --cluster_name "b2"


CUDA_VISIBLE_DEVICES=3 python main_pretrain.py --seed 0 --alpha_entropy 0 --alpha_sparsity 1. --batch_size 256 --classifier_lr 0.1 --dataset "ecg-TCH-40_patient-20220201" --lr 0.13234838295784523 --mask_lr 0.01946841419435407 --optimizer sgd --scheduler warmup_cosine --weight_decay 8.504658547335148e-04 --temperature 0.2 --max_epochs 400 --N 2 --n_crops 1 --n_small_crops 0 --encoder resnet1d --mask_fbase 128 --method simclr_adios_1d --output_dim 256 --proj_hidden_dim 2048 --unet_norm gn --data_dir "/mnt/group1/yilong/JET-Detection-Data" --wandb_dir "/home/yilong/Github/adios" --checkpoint_dir "/home/yilong/Github/adios/trained_models" --project "adios_ecg_cluster_debug_Nmask2" --entity "yilongju" --name "simclr_adios_resnet18_ECG_cluster_debug_Nmask2_10x_more_decay" --wandb True --ptl_accelerator "ddp" --gpus 0 --cluster_name "b2"

