# Windows
python main_finetune.py --dataset "ecg-TCH-40_patient-20220201" --data_dir "D:\\Dropbox\\Study\\GitHub\\adios\\data" --max_epochs 2 --gpus 0 --precision 32 --optimizer sgd --scheduler warmup_cosine --lr 0.5 --weight_decay 5e-4 --batch_size 256 --num_workers 4 --name "finetune_resnet18_ECG_debug" --project "adios_ecg_debug" --entity "yilongju" --wandb True --ptl_accelerator "cpu" --debug True

# MacOS
python main_finetune.py --seed 0 --encoder "resnet1d" --dataset "ecg-TCH-40_patient-20220201" --data_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data" --max_epochs 2 --precision 32 --optimizer sgd --scheduler warmup_cosine --lr 0.5 --weight_decay 5e-4 --batch_size 256 --num_workers 0 --checkpoint_dir "/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models"  --project "adios_ecg_debug" --entity "yilongju" --name "finetune_resnet18_ECG_debug" --wandb True --ptl_accelerator "cpu" --read_data_by_chunk False --debug True
