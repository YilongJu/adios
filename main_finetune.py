import os
import json
from pathlib import Path
from torch import nn

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from src.args.setup import parse_args_finetune
from src.args.utils import IMG_SIZE_DATASET

try:
    from src.methods.dali import ClassificationABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True
from src.methods.supervised import SupervisedModel
from src.methods.supervised_1d import SupervisedModel_1D
from src.utils.classification_dataloader import prepare_data
from src.utils.checkpointer import Checkpointer
from src.methods import METHODS
from src.methods.base import SUPPORTED_NETWORKS

from src.utils.ECG_data_loading import *
from src.utils.pretrain_dataloader import prepare_dataloader


def main():
    args = parse_args_finetune()
    if args.mask_feature_extractor is not None:
        # build paths
        ckpt_dir = Path(args.mask_feature_extractor)
        args_path = ckpt_dir / "args.json"
        ckpt_path_list = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")]
        if args.ckpt_epoch >= 0:
            ckpt_path_list = [ele for ele in ckpt_path_list if f"ep={args.ckpt_epoch}.ckpt" in str(ele)]
        print(f"[Filtered] ckpt_path_list: {ckpt_path_list}")
        ckpt_path = ckpt_path_list[0]

        # load arguments
        with open(args_path) as f:
            method_args = json.load(f)

        # build the model
        model_base = METHODS[method_args["method"]].load_from_checkpoint(
            ckpt_path, strict=False, **method_args
        )
        if args.precision == 32:
            pretrained_occlusion_model_dict = {"mask_encoder": model_base.mask_encoder.float(),
                                               "mask_head": model_base.mask_head.float()}
        elif args.precision == 16:
            pretrained_occlusion_model_dict = {"mask_encoder": model_base.mask_encoder.half(),
                                               "mask_head": model_base.mask_head.half()}
        elif args.precision == 64:
            pretrained_occlusion_model_dict = {"mask_encoder": model_base.mask_encoder.double(),
                                               "mask_head": model_base.mask_head.double()}
        else:
            raise NotImplementedError("Unknown precision.")

    else:
        pretrained_occlusion_model_dict = None

    if args.pretrained_feature_extractor is not None:
        # build paths
        ckpt_dir = Path(args.pretrained_feature_extractor)
        args_path = ckpt_dir / "args.json"
        ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][0]

        # load arguments
        with open(args_path) as f:
            method_args = json.load(f)

        # build the model
        model_base = METHODS[method_args["method"]].load_from_checkpoint(
            ckpt_path, strict=False, **method_args
        )
        model = model_base.encoder

    else:
        base_model = SUPPORTED_NETWORKS[args.encoder]
        model = base_model(zero_init_residual=args.zero_init_residual, embedding_dim=args.embedding_dim)
        # remove fc layer
        model.fc = nn.Identity()

    model.pretrained_occlusion_model_dict = pretrained_occlusion_model_dict

    # model = SupervisedModel(model, **args.__dict__)

    if args.dataset in ["ecg-TCH-40_patient-20220201"]:
        model = SupervisedModel_1D(model, **args.__dict__)

        feature_with_ecg_df_train, feature_with_ecg_df_test, feature_with_ecg_df_dev, feature_with_ecg_df_val, save_folder = Data_preprocessing(
            args)
        channel_ID = args.channel_ID
        """ Get dataloader """
        feature_with_ecg_df_train_single_lead = feature_with_ecg_df_dev.query(f"channel_ID == {channel_ID}")
        feature_with_ecg_df_test_single_lead = feature_with_ecg_df_val.query(f"channel_ID == {channel_ID}")

        ecg_resampling_length = args.ecg_resampling_length
        ecg_colnames = [f"ecg{i + 1}" for i in range(ecg_resampling_length)]
        ecg_mat = feature_with_ecg_df_train_single_lead[ecg_colnames].values
        signal_min_train = np.min(ecg_mat.ravel())

        train_dataset = ECG_classification_dataset_with_peak_features(feature_with_ecg_df_train_single_lead,
                                                                      shift_signal=args.shift_signal,
                                                                      shift_amount=signal_min_train,
                                                                      normalize_signal=args.normalize_signal,
                                                                      ecg_resampling_length_target=args.ecg_resampling_length_target,
                                                                      transforms=args.transforms)
        test_dataset = ECG_classification_dataset_with_peak_features(feature_with_ecg_df_test_single_lead,
                                                                     shift_signal=args.shift_signal,
                                                                     shift_amount=signal_min_train,
                                                                     normalize_signal=args.normalize_signal,
                                                                     ecg_resampling_length_target=args.ecg_resampling_length_target,
                                                                     transforms=args.transforms)

        if Lower(args.transforms) == Lower("Identity"):
            args.batch_size = args.batch_size * 2

        train_loader = prepare_dataloader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True
        )
        val_loader = prepare_dataloader(
            test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False
        )
    else:
        model = SupervisedModel(model, **args.__dict__)

        train_loader, val_loader = prepare_data(
            dataset=args.dataset,
            size=IMG_SIZE_DATASET[args.dataset],
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    callbacks = []

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name, project=args.project, entity=args.entity, offline=args.offline
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, "linear"),
            frequency=args.checkpoint_frequency,
            keep_previous_checkpoints=True
        )
        callbacks.append(ckpt)

    print(
        args)  # Namespace(accelerator=None, accumulate_grad_batches=1, amp_backend='native', amp_level='O2', auto_lr_find=False, auto_scale_batch_size=False, auto_select_gpus=False, batch_size=256, benchmark=False, channel_ID=2, check_val_every_n_epoch=1, checkpoint_callback=True, checkpoint_dir='/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models', checkpoint_frequency=1, cifar=False, classifier_lr=0.3, cluster_name='b4', dali=False, dali_device='gpu', data_chunk_folder='ecg-pat40-tch-sinus_jet', data_dir='/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data', dataset='ecg-TCH-40_patient-20220201', debug=True, default_root_dir=None, deterministic=False, devices=None, distributed_backend=None, ecg_resampling_length=300, encoder='resnet18', entity='yilongju', exclude_bias_n_norm=True, extra_optimizer_args={'momentum': 0.9}, fast_dev_run=False, flush_logs_every_n_steps=100, gpus=[0], gradient_clip_algorithm='norm', gradient_clip_val=0.0, ipus=None, lars=True, limit_predict_batches=1.0, limit_test_batches=1.0, limit_train_batches=1.0, limit_val_batches=1.0, log_every_n_steps=50, log_gpu_memory=None, logger=True, lr=0.5, lr_decay_steps=None, max_epochs=2, max_steps=None, max_time=None, min_epochs=None, min_steps=None, move_metrics_to_cpu=False, multiple_trainloader_mode='max_size_cycle', n_classes=2, name='finetune_resnet18_ECG_debug', no_labels=True, normalize_signal=False, num_nodes=1, num_processes=1, num_sanity_val_steps=2, num_workers=4, offline=False, optimizer='sgd', overfit_batches=0.0, plugins=None, precision=16, prepare_data_per_node=True, pretrained_feature_extractor=None, process_position=0, profiler=None, progress_bar_refresh_rate=None, project='adios_ecg_debug', ptl_accelerator='cpu', read_data_by_chunk=True, reload_dataloaders_every_epoch=False, reload_dataloaders_every_n_epochs=0, replace_sampler_ddp=True, resume_from_checkpoint=None, scheduler='warmup_cosine', seed=-1, shift_signal=False, stochastic_weight_avg=False, sync_batchnorm=False, target_type='single', terminate_on_nan=False, tpu_cores=None, track_grad_norm=-1, train_dir=None, truncated_bptt_steps=None, use_mask=False, val_check_interval=1.0, val_dir=None, validation_frequency=1, wandb=True, weight_decay=0.0005, weights_save_path=None, weights_summary='top', zero_init_residual=None)

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        plugins=None if args.ptl_accelerator in ["cpu"] else DDPPlugin(find_unused_parameters=False),
        checkpoint_callback=False,
        terminate_on_nan=True,
        accelerator=args.ptl_accelerator,
        devices=1 if args.ptl_accelerator in ["cpu"] else None,
        check_val_every_n_epoch=args.validation_frequency,
        deterministic=False if args.ptl_accelerator in ["cpu"] else True
    )
    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
