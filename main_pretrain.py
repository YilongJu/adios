import os, sys
import numpy as np
from pprint import pprint

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from src.args.setup import parse_args_pretrain
from src.methods import METHODS
from src.utils.auto_resumer import AutoResumer
from src.utils.pretrain_dataloader import dataset_with_index

try:
    from src.methods.dali import PretrainABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

from src.utils.auto_mask import AutoMASK
try:
    from src.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

from src.utils.checkpointer import Checkpointer
from src.utils.classification_dataloader import prepare_data as prepare_data_classification
from src.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_multicrop_transform,
    prepare_n_crop_transform,
    prepare_transform,
)

from src.utils.ECG_data_loading import *


def main():
    # seed = np.random.randint(0, 2**32)
    # seed_everything(seed)
    args = parse_args_pretrain()
    seed = args.seed
    if seed >= 0:
        seed_everything(seed, workers=True)
    else:
        seed = np.random.randint(0, 2 ** 32)
        seed_everything(seed, workers=True)


    if sys.gettrace() is not None:
        args.num_workers = 0

    assert args.method in METHODS, f"Choose from {METHODS.keys()}"

    MethodClass = METHODS[args.method]
    if args.dali:
        print("\n======== Using dali... ========")
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with [dali]."
        MethodClass = type(f"Dali{MethodClass.__name__}", (MethodClass, PretrainABC), {})
    model = MethodClass(**args.__dict__)

    if args.dataset in ["ecg-TCH-40_patient-20220201"]:
        feature_with_ecg_df_train, feature_with_ecg_df_test, save_folder = Data_preprocessing(args)
        channel_ID = args.channel_ID
        """ Get dataloader """
        feature_with_ecg_df_train_single_lead = feature_with_ecg_df_train.query(f"channel_ID == {channel_ID}")
        feature_with_ecg_df_test_single_lead = feature_with_ecg_df_test.query(f"channel_ID == {channel_ID}")

        ecg_resampling_length = args.ecg_resampling_length
        ecg_colnames = [f"ecg{i + 1}" for i in range(ecg_resampling_length)]
        ecg_mat = feature_with_ecg_df_train_single_lead[ecg_colnames].values
        signal_min_train = np.min(ecg_mat.ravel())

        train_dataset = dataset_with_index(ECG_classification_dataset_with_peak_features)(feature_with_ecg_df_train_single_lead, shift_signal=args.shift_signal, shift_amount=signal_min_train, normalize_signal=args.normalize_signal, ecg_resampling_length_target=args.ecg_resampling_length_target)
        test_dataset = ECG_classification_dataset_with_peak_features(feature_with_ecg_df_test_single_lead, shift_signal=args.shift_signal, shift_amount=signal_min_train, normalize_signal=args.normalize_signal, ecg_resampling_length_target=args.ecg_resampling_length_target)

        train_loader = prepare_dataloader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True
        )
        val_loader = prepare_dataloader(
            test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False
        )
    else:
        # add img size to transform kwargs
        args.transform_kwargs.update({"size": args.img_size})

        # contrastive dataloader
        if not args.dali:
            print("\n======== Not using dali... ========")
            # asymmetric augmentations
            if args.unique_augs > 1:
                print("----- args.unique_augs > 1 -----")
                transform = [
                    prepare_transform(args.dataset, multicrop=args.multicrop, **kwargs)
                    for kwargs in args.transform_kwargs
                ]
            else:
                print("----- not args.unique_augs > 1 -----")
                transform = prepare_transform(
                    args.dataset, multicrop=args.multicrop, **args.transform_kwargs
                )

            if args.debug_augmentations:
                print("----- args.debug_augmentations -----")
                print("Transforms:")
                pprint(transform)

            if args.multicrop:
                print("----- args.multicrop -----")
                assert not args.unique_augs == 1

                if args.dataset in ["cifar10", "cifar100"]:
                    size_crops = [32, 24]
                elif args.dataset == "stl10":
                    size_crops = [96, 58]
                # imagenet or custom dataset
                else:
                    size_crops = [224, 96]

                transform = prepare_multicrop_transform(
                    transform, size_crops=size_crops, n_crops=[args.n_crops, args.n_small_crops]
                )
            else:
                print("----- not args.multicrop -----")
                if args.n_crops != 2:
                    assert args.method == "wmse"
                transform = prepare_n_crop_transform(transform, n_crops=args.n_crops)

            train_dataset = prepare_datasets(
                args.dataset,
                transform,
                data_dir=args.data_dir,
                train_dir=args.train_dir,
                morphology=args.morph,
                load_masks=args.load_masks,
            )
            print(f"train_dataset size = {len(train_dataset)}")
            train_loader = prepare_dataloader(
                train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
            )
            print(f"train_loader size = # batches {len(train_loader)} * batch_size {args.batch_size} = {len(train_loader) * args.batch_size}")

        # normal dataloader for when it is available
        _, val_loader = prepare_data_classification(
            args.dataset,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            size=args.img_size,
            load_masks=args.load_masks,
        )

        # print(f"args = {args}") # args = Namespace(N=6, accelerator=None, accumulate_grad_batches=1, alpha_entropy=0.0, alpha_sparsity=0.0, amp_backend='native', amp_level='O2', auto_lr_find=False, auto_mask=True, auto_mask_dir='auto_mask', auto_mask_frequency=1, auto_resume=False, auto_scale_batch_size=False, auto_select_gpus=False, auto_umap=False, batch_size=256, benchmark=False, brightness=[0.8], check_val_every_n_epoch=1, checkpoint_callback=True, checkpoint_dir='/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/trained_models', checkpoint_frequency=1, cifar=False, classifier_lr=0.1, contrast=[0.8], dali=False, dali_device='gpu', data_dir='/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios/data', dataset='stl10', debug_augmentations=None, default_root_dir=None, deterministic=False, devices=None, disable_knn_eval=True, distributed_backend=None, encoder='resnet18', entity='yilongju', eta_lars=0.02, exclude_bias_n_norm=True, extra_optimizer_args={'momentum': 0.9}, fast_dev_run=False, flush_logs_every_n_steps=100, gaussian_prob=[0.5], gpus=None, grad_clip_lars=True, gradient_clip_algorithm='norm', gradient_clip_val=0.0, hue=[0.2], img_size=96, ipus=None, knn_k=20, lars=True, limit_predict_batches=1.0, limit_test_batches=1.0, limit_train_batches=1.0, limit_val_batches=1.0, load_masks=True, log_every_n_steps=50, log_gpu_memory=None, logger=True, lr=0.13234838295784523, lr_decay_steps=None, mask_fbase=128, mask_lr=0.01946841419435407, max_epochs=400, max_steps=None, max_time=None, mean=[0.485, 0.456, 0.406], method='simclr_adios', min_epochs=None, min_lr=0.0, min_scale=[0.08], min_steps=None, morph='none', move_metrics_to_cpu=False, multicrop=None, multiple_trainloader_mode='max_size_cycle', n_classes=10, n_crops=2, n_small_crops=0, name='simclr_adios_resnet18_stl10_debug', nice=False, no_labels=True, num_nodes=1, num_processes=1, num_sanity_val_steps=2, num_workers=5, offline=False, optimizer='sgd', output_dim=256, overfit_batches=0.0, plugins=None, precision=32, prepare_data_per_node=True, pretrained_dir=None, process_position=0, profiler=None, progress_bar_refresh_rate=None, proj_hidden_dim=2048, project='adios_debug', ptl_accelerator='cpu', reload_dataloaders_every_epoch=False, reload_dataloaders_every_n_epochs=0, replace_sampler_ddp=True, resume_from_checkpoint=None, saturation=[0.8], scheduler='warmup_cosine', seed=-1, size=[224], solarization_prob=[0.0], std=[0.228, 0.224, 0.225], stochastic_weight_avg=False, sync_batchnorm=False, target_type='single', temperature=0.2, terminate_on_nan=False, tpu_cores=None, track_grad_norm=-1, train_dir=None, train_mask_epoch=0, transform_kwargs={'brightness': 0.8, 'contrast': 0.8, 'saturation': 0.8, 'hue': 0.2, 'gaussian_prob': 0.5, 'solarization_prob': 0.0, 'min_scale': 0.08, 'size': 96}, truncated_bptt_steps=None, unet_norm='gn', unique_augs=1, val_check_interval=1.0, val_dir=None, validation_frequency=1, wandb=True, wandb_dir='/Users/yj31/Dropbox/My Mac (C02FR2BBMD6T)/Documents/GitHub/adios', warmup_epochs=10, warmup_start_lr=0.003, weight_decay=8.504658547335148e-05, weights_save_path=None, weights_summary='top', zero_init_residual=None)

    """ Dataloader debug """
    print("Train dataloader")
    for i, batch in enumerate(train_loader):
        print(i, len(batch))
        for ele in batch:
            try:
                print(f"ele.shape: {ele.shape}")
                # print(f"{ele[:5, ...]}")
            except:
                print(f"ele.len: {len(ele)}")
                for ele2 in ele:
                    print(f"ele.shape: {ele2.shape}")
                    # print(f"{ele2[:5, ...]}")
        if i >= 1:
            break

    print("Val dataloader")
    for i, batch in enumerate(val_loader):
        print(i, len(batch))
        for ele in batch:
            try:
                print(f"ele.shape: {ele.shape}")
                # print(f"{ele[:5, ...]}")
            except:
                print(f"ele.len: {len(ele)}")
                for ele2 in ele:
                    print(f"ele.shape: {ele2.shape}")
                    # print(f"{ele2[:5, ...]}")
        if i >= 1:
            break



    callbacks = []

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name,
            project=args.project,
            entity=args.entity,
            offline=args.offline,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # save checkpoint on last epoch only [all / epochs]
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, args.method),
            frequency=args.checkpoint_frequency,
            keep_previous_checkpoints=True
        )
        callbacks.append(ckpt)

        if args.auto_umap:
            assert (
                _umap_available
            ), "UMAP is not currently avaiable, please install it first with [umap]."
            auto_umap = AutoUMAP(
                args,
                logdir=os.path.join(args.auto_umap_dir, args.method),
                frequency=args.auto_umap_frequency,
            )
            callbacks.append(auto_umap)

        if args.auto_mask:
            auto_mask = AutoMASK(
                args,
                logdir=os.path.join(args.auto_mask_dir, args.method),
                frequency=args.auto_mask_frequency,
            )
            callbacks.append(auto_mask)

    if args.auto_resume and args.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(args.checkpoint_dir, args.method),
            load_dir=args.load_dir,
            search_in_checkpoint_dir=args.search_in_checkpoint_dir
        )
        resume_from_checkpoint = auto_resumer.find_checkpoint(args)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            args.resume_from_checkpoint = resume_from_checkpoint


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

