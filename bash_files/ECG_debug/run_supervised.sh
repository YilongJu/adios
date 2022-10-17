# Windows
python main_finetune.py --dataset "ecg-TCH-40_patient-20220201" --data_dir "D:\\Dropbox\\Study\\GitHub\\adios\\data" --max_epochs 2 --gpus 0 --precision 32 --optimizer sgd --scheduler warmup_cosine --lr 0.5 --weight_decay 5e-4 --batch_size 256 --num_workers 4 --name "finetune_resnet18_ECG_debug" --project "adios_ecg_debug" --entity "yilongju" --wandb True --ptl_accelerator "cpu" --debug True

# MacOS
python main_finetune.py --seed 0 --encoder "resnet1d" --dataset "ecg-TCH-40_patient-20220201" --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --max_epochs 2 --precision 32 --optimizer sgd --scheduler warmup_cosine --classifier_lr 0.001 --lr 0.001 --weight_decay 5e-4 --batch_size 256 --num_workers 0 --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models"  --project "adios_ecg_debug" --entity "yilongju" --name "finetune_resnet18_ECG_debug" --wandb True --ptl_accelerator "cpu" --train_backbone True --read_data_by_chunk False --debug True
python main_finetune.py --seed 0 --encoder "resnet1d" --dataset "ecg-TCH-40_patient-20220201" --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --max_epochs 3 --precision 32 --optimizer sgd --scheduler warmup_cosine --lr 0.001 --weight_decay 5e-4 --batch_size 256 --num_workers 0 --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models"  --project "adios_ecg_debug" --entity "yilongju" --name "finetune_resnet18_ECG_debug" --wandb True --ptl_accelerator "cpu" --train_backbone True --read_data_by_chunk False --debug True
python main_finetune.py --seed 0 --encoder "clocs_cnn1d" --ecg_resampling_length_target 2500 --embedding_dim 256 --normalize_signal True --dataset "ecg-TCH-40_patient-20220201" --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --max_epochs 2 --precision 32 --optimizer sgd --scheduler warmup_cosine --classifier_lr 0.001 --lr 0.001 --weight_decay 5e-4 --batch_size 256 --num_workers 0 --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models"  --project "adios_ecg_debug" --entity "yilongju" --name "supervised_clocs_ECG_debug" --wandb True --ptl_accelerator "cpu" --train_backbone True --read_data_by_chunk False --debug True
python main_finetune.py --seed 0 --encoder "clocs_cnn1d" --ecg_resampling_length_target 2500 --embedding_dim 64 --normalize_signal True --dataset "ecg-TCH-40_patient-20220201" --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --max_epochs 2 --precision 32 --optimizer sgd --scheduler warmup_cosine --classifier_lr 0.001 --lr 0.001 --weight_decay 5e-4 --batch_size 256 --num_workers 0 --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models"  --project "adios_ecg_debug" --entity "yilongju" --name "supervised_clocs_ECG_debug" --wandb True --ptl_accelerator "cpu" --train_backbone True --read_data_by_chunk False --debug True
python main_finetune.py --seed 0 --transforms "gaussian" --encoder "clocs_cnn1d" --ecg_resampling_length_target 2500 --embedding_dim 64 --normalize_signal True --dataset "ecg-TCH-40_patient-20220201" --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --max_epochs 2 --precision 32 --optimizer sgd --scheduler warmup_cosine --classifier_lr 0.001 --lr 0.001 --weight_decay 5e-4 --batch_size 256 --num_workers 0 --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models"  --project "adios_ecg_debug" --entity "yilongju" --name "supervised_clocs_ECG_debug" --wandb True --ptl_accelerator "cpu" --train_backbone True --read_data_by_chunk False --debug True
python main_finetune.py --seed 0 --transforms "gaussian" --encoder "clocs_cnn1d" --ecg_resampling_length_target 2500 --embedding_dim 64 --normalize_signal True --dataset "ecg-TCH-40_patient-20220201" --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --max_epochs 4 --precision 32 --optimizer sgd --scheduler warmup_cosine --classifier_lr 0.001 --lr 0.001 --weight_decay 5e-4 --batch_size 256 --num_workers 0 --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models"  --project "adios_ecg_debug" --entity "yilongju" --name "supervised_clocs_ECG_debug" --wandb True --ptl_accelerator "cpu" --train_backbone True --read_data_by_chunk False --debug True

python main_finetune.py --seed 0 --transforms "gaussian" --encoder "clocs_cnn1d" --ecg_resampling_length_target 2500 --embedding_dim 64 --normalize_signal True --dataset "ecg-TCH-40_patient-20220201" --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --max_epochs 2 --precision 32 --optimizer adam --scheduler warmup_cosine --classifier_lr 0.001 --lr 0.001 --weight_decay 5e-4 --batch_size 256 --num_workers 0 --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models"  --project "adios_ecg_debug" --entity "yilongju" --name "supervised_clocs_ECG_debug" --wandb True --ptl_accelerator "cpu" --train_backbone True --read_data_by_chunk False --debug True
python main_finetune.py --seed 0 --transforms "identity" --encoder "clocs_cnn1d" --ecg_resampling_length_target 2500 --embedding_dim 64 --normalize_signal True --dataset "ecg-TCH-40_patient-20220201" --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --max_epochs 2 --precision 32 --optimizer adam --scheduler warmup_cosine --classifier_lr 0.001 --lr 0.001 --weight_decay 5e-4 --batch_size 256 --num_workers 0 --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models"  --project "adios_ecg_debug" --entity "yilongju" --name "supervised_clocs_ECG_debug" --wandb True --ptl_accelerator "cpu" --train_backbone True --read_data_by_chunk False --debug True

## ECG300
python main_finetune.py --seed 0 --c4_multiplier 3 --transforms "gaussian" --encoder "clocs_cnn1d" --ecg_resampling_length_target 300 --stride 2 --embedding_dim 64 --normalize_signal True --dataset "ecg-TCH-40_patient-20220201" --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --max_epochs 2 --precision 32 --optimizer adam --scheduler warmup_cosine --classifier_lr 0.001 --lr 0.001 --weight_decay 5e-4 --batch_size 256 --num_workers 0 --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models"  --project "adios_ecg_debug" --entity "yilongju" --name "supervised_clocs_ECG_debug" --wandb True --ptl_accelerator "cpu" --train_backbone True --read_data_by_chunk False --debug True
python main_finetune.py --seed 0 --wandb_dir "wandb_test" --c4_multiplier 3 --transforms "gaussian" --encoder "clocs_cnn1d" --ecg_resampling_length_target 300 --stride 2 --embedding_dim 64 --normalize_signal True --dataset "ecg-TCH-40_patient-20220201" --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --max_epochs 2 --precision 32 --optimizer adam --scheduler warmup_cosine --classifier_lr 0.001 --lr 0.001 --weight_decay 5e-4 --batch_size 256 --num_workers 0 --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models"  --project "adios_ecg_debug" --entity "yilongju" --name "supervised_clocs_ECG_debug" --wandb True --ptl_accelerator "cpu" --train_backbone True --read_data_by_chunk False --debug True
python main_finetune.py --seed 0 --wandb_dir "wandb_test" --c4_multiplier 3 --transforms "longitudinal" --encoder "clocs_cnn1d" --ecg_resampling_length_target 300 --stride 2 --embedding_dim 64 --normalize_signal True --dataset "ecg-TCH-40_patient-20220201" --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --max_epochs 2 --precision 32 --optimizer adam --scheduler warmup_cosine --classifier_lr 0.001 --lr 0.001 --weight_decay 5e-4 --batch_size 256 --num_workers 0 --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models"  --project "adios_ecg_debug" --entity "yilongju" --name "supervised_clocs_ECG_debug" --wandb True --ptl_accelerator "cpu" --train_backbone True --read_data_by_chunk False --debug True


# Linux
python main_finetune.py --seed 0 --transforms "gaussian" --encoder "clocs_cnn1d" --ecg_resampling_length_target 2500 --embedding_dim 64 --normalize_signal True --dataset "ecg-TCH-40_patient-20220201" --max_epochs 2 --precision 32 --optimizer sgd --scheduler warmup_cosine --classifier_lr 0.001 --lr 0.001 --weight_decay 5e-4 --batch_size 256 --num_workers 0 --project "adios_ecg_debug" --entity "yilongju" --name "supervised_clocs_ECG_debug" --wandb True --ptl_accelerator "cpu" --train_backbone True --read_data_by_chunk False --debug True
python main_finetune.py --seed 0 --transforms "gaussian" --encoder "clocs_cnn1d" --ecg_resampling_length_target 2500 --embedding_dim 64 --normalize_signal True --dataset "ecg-TCH-40_patient-20220201" --max_epochs 2 --precision 32 --optimizer sgd --scheduler warmup_cosine --classifier_lr 0.001 --lr 0.001 --weight_decay 5e-4 --batch_size 256 --num_workers 0 --project "adios_ecg_debug" --entity "yilongju" --name "supervised_clocs_ECG_debug" --wandb True --ptl_accelerator "ddp" --gpus 0 --train_backbone True --read_data_by_chunk False --debug True
python main_finetune.py --seed 0 --transforms "gaussian" --encoder "clocs_cnn1d" --ecg_resampling_length_target 2500 --embedding_dim 64 --normalize_signal True --dataset "ecg-TCH-40_patient-20220201" --max_epochs 2 --precision 32 --optimizer sgd --scheduler warmup_cosine --classifier_lr 0.001 --lr 0.001 --weight_decay 5e-4 --batch_size 256 --num_workers 4 --project "adios_ecg_debug" --entity "yilongju" --name "supervised_clocs_ECG_debug" --wandb True --ptl_accelerator "ddp" --gpus 0 --train_backbone True --read_data_by_chunk False --debug True
python main_finetune.py --seed 0 --transforms "gaussian" --encoder "clocs_cnn1d" --ecg_resampling_length_target 2500 --embedding_dim 64 --normalize_signal True --dataset "ecg-TCH-40_patient-20220201" --max_epochs 2 --precision 32 --optimizer sgd --scheduler warmup_cosine --classifier_lr 0.001 --lr 0.001 --weight_decay 5e-4 --batch_size 256 --num_workers 4 --project "adios_ecg_debug" --entity "yilongju" --name "supervised_clocs_ECG_debug" --wandb True --ptl_accelerator "ddp" --gpus 0,1,2,3 --train_backbone True --read_data_by_chunk True --cluster_name "auto"


# MacOS + pretrained_mask
python main_finetune.py --seed 0 --encoder "resnet1d" --dataset "ecg-TCH-40_patient-20220201" --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --max_epochs 2 --precision 32 --optimizer sgd --scheduler warmup_cosine --lr 0.001 --weight_decay 5e-4 --batch_size 256 --num_workers 0 --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models"  --project "adios_ecg_debug" --entity "yilongju" --name "finetune_resnet18_ECG_debug" --wandb True --ptl_accelerator "cpu" --train_backbone True --mask_feature_extractor "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models/simclr_adios_1d/fqbfkouz" --ckpt_epoch 0 --read_data_by_chunk False --debug True
