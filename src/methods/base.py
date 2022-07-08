from argparse import ArgumentParser
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple
from torchvision.models import resnet18, resnet50

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.lars import LARSWrapper
from src.utils.metrics import accuracy_at_k, weighted_mean, multiclass_accuracy, compute_auroc, AUROC
from src.utils.momentum import MomentumUpdater, initialize_momentum_params
from src.utils.knn import WeightedKNNClassifier
from src.utils.blocks import str2bool
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from src.utils.backbones import (
    poolformer_m36,
    poolformer_m48,
    poolformer_s12,
    poolformer_s24,
    poolformer_s36,
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
)

from src.models.ResNet1D import ResNet1D
import numpy as np

SUPPORTED_NETWORKS = {
            "resnet18": resnet18,
            "resnet50": resnet50,
            "vit_tiny": vit_tiny,
            "vit_small": vit_small,
            "vit_base": vit_base,
            "vit_large": vit_large,
            "swin_tiny": swin_tiny,
            "swin_small": swin_small,
            "swin_base": swin_base,
            "swin_large": swin_large,
            "poolformer_s12": poolformer_s12,
            "poolformer_s24": poolformer_s24,
            "poolformer_s36": poolformer_s36,
            "poolformer_m36": poolformer_m36,
            "poolformer_m48": poolformer_m48,
            "resnet1d": ResNet1D,
        }

def static_lr(
    get_lr: Callable, param_group_indexes: Sequence[int], lrs_to_replace: Sequence[float]
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs

softmax = torch.nn.Softmax(dim=1)

class BaseModel(pl.LightningModule):
    def __init__(
        self,
        encoder: str,
        n_classes: int,
        target_type: str,
        cifar: bool,
        zero_init_residual: bool,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lars: bool,
        lr: float,
        weight_decay: float,
        classifier_lr: float,
        exclude_bias_n_norm: bool,
        accumulate_grad_batches: int,
        extra_optimizer_args: Dict,
        scheduler: str,
        min_lr: float,
        warmup_start_lr: float,
        warmup_epochs: float,
        multicrop: bool,
        n_crops: int,
        n_small_crops: int,
        img_size: int,
        eta_lars: float = 1e-3,
        grad_clip_lars: bool = False,
        lr_decay_steps: Sequence = None,
        disable_knn_eval: bool = True,
        knn_k: int = 20,
        **kwargs,
    ):
        """Base model that implements all basic operations for all self-supervised methods.
        It adds shared arguments, extract basic learnable parameters, creates optimizers
        and schedulers, implements basic training_step for any number of crops,
        trains the online classifier and implements validation_step.

        Args:
            encoder (str): architecture of the base encoder.
            n_classes (int): number of classes.
            cifar (bool): flag indicating if cifar is being used.
            zero_init_residual (bool): change the initialization of the resnet encoder.
            max_epochs (int): number of training epochs.
            batch_size (int): number of samples in the batch.
            optimizer (str): name of the optimizer.
            lars (bool): flag indicating if lars should be used.
            lr (float): learning rate.
            weight_decay (float): weight decay for optimizer.
            classifier_lr (float): learning rate for the online linear classifier.
            exclude_bias_n_norm (bool): flag indicating if bias and norms should be excluded from
                lars.
            accumulate_grad_batches (int): number of batches for gradient accumulation.
            extra_optimizer_args (Dict): extra named arguments for the optimizer.
            scheduler (str): name of the scheduler.
            min_lr (float): minimum learning rate for warmup scheduler.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
            warmup_epochs (float): number of warmup epochs.
            multicrop (bool): flag indicating if multi-resolution crop is being used.
            n_crops (int): number of big crops
            n_small_crops (int): number of small crops (will be set to 0 if multicrop is False).
            eta_lars (float): eta parameter for lars.
            grad_clip_lars (bool): whether to clip the gradients in lars.
            lr_decay_steps (Sequence, optional): steps to decay the learning rate if scheduler is
                step. Defaults to None.
            disable_knn_eval (bool): disables online knn evaluation while training.
            knn_k (int): the number of neighbors to use for knn.
        """

        super().__init__()

        # back-bone related
        self.cifar = cifar
        self.zero_init_residual = zero_init_residual

        # training related
        self.n_classes = n_classes
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lars = lars
        self.lr = lr
        self.weight_decay = weight_decay
        self.classifier_lr = classifier_lr
        self.exclude_bias_n_norm = exclude_bias_n_norm
        self.accumulate_grad_batches = accumulate_grad_batches
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.lr_decay_steps = lr_decay_steps
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.multicrop = multicrop
        self.n_crops = n_crops
        self.n_small_crops = n_small_crops
        self.eta_lars = eta_lars
        self.grad_clip_lars = grad_clip_lars
        self.img_size = img_size
        self.disable_knn_eval = disable_knn_eval
        self.knn_k = knn_k

        """ member for auroc calculation """
        self.train_auroc = AUROC(pos_label=1)
        self.val_auroc = AUROC(pos_label=1)

        # sanity checks on multicrop
        if self.multicrop:
            assert n_small_crops > 0
        else:
            self.n_small_crops = 0

        # all the other parameters
        self.extra_args = kwargs

        # if accumulating gradient then scale lr
        self.lr = self.lr * self.accumulate_grad_batches
        self.classifier_lr = self.classifier_lr * self.accumulate_grad_batches
        self.min_lr = self.min_lr * self.accumulate_grad_batches
        self.warmup_start_lr = self.warmup_start_lr * self.accumulate_grad_batches

        assert encoder in SUPPORTED_NETWORKS.keys()
        self.base_model = SUPPORTED_NETWORKS[encoder]
        self.encoder_name = encoder

        # initialize encoder
        self.encoder_kwargs = {"zero_init_residual": zero_init_residual} \
            if 'resnet' in encoder else {"img_size": img_size}
        self.encoder = self.base_model(**self.encoder_kwargs).to(torch.float)

        print(f"self.n_crops = {self.n_crops}")
        print(f"self.n_small_crops = {self.n_small_crops}")

        if "resnet" in encoder:
            self.features_size = self.encoder.inplanes
            if "1d" in encoder:
                # self.features_size *= 128 # For the original (unofficial) ResNet 1D
                pass
            # remove fc layer
            self.encoder.fc = nn.Identity()
            if cifar:
                self.encoder.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
                self.encoder.maxpool = nn.Identity()
        else:
            self.features_size = self.encoder.num_features

        self.classifier = nn.Linear(self.features_size, n_classes)

        # linear classifier metrics
        self.metric_keys = ['acc1', 'acc5'] if target_type == 'single' \
            else ['f1_macro', 'f1_micro', 'f1_weighted']
        self.target_type = target_type
        if target_type == 'multi':
            self.loss_fn = F.binary_cross_entropy_with_logits
            self.metric_fn = multiclass_accuracy
            self.loss_args, self.metric_args = {}, {}
        elif target_type == "single":
            self.loss_fn = F.cross_entropy
            self.metric_fn = accuracy_at_k
            self.loss_args, self.metric_args = {"ignore_index": -1}, {"top_k": (1, min(5, n_classes))}
            if "1d" in encoder:
                self.metric_keys = ['acc1']
                self.metric_args = {"top_k": (1,)}

        if not self.disable_knn_eval:
            self.knn = WeightedKNNClassifier(k=self.knn_k, distance_fx="euclidean")

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds shared basic arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("base")
        parser.add_argument("--nice", type=str2bool, nargs='?',
                            const=True, default=False,
                            help="Activate nice mode.")
        # encoder args
        parser.add_argument("--encoder", choices=SUPPORTED_NETWORKS.keys(), type=str, default='resnet18')
        parser.add_argument("--zero_init_residual", type=str2bool, nargs='?',
                            const=True)

        # general train
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--lr", type=float, default=0.4)
        parser.add_argument("--classifier_lr", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        parser.add_argument("--num_workers", type=int, default=5)

        # wandb
        parser.add_argument("--name", type=str)
        parser.add_argument("--project", type=str)
        parser.add_argument("--entity", type=str)
        parser.add_argument("--wandb", type=str2bool, nargs='?',
                            const=True, default=False)
        parser.add_argument("--offline", type=str2bool, nargs='?',
                            const=True, default=False)

        # optimizer
        SUPPORTED_OPTIMIZERS = ["sgd", "adam"]

        parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS, type=str, default='sgd')
        parser.add_argument("--lars", type=str2bool, nargs='?',
                            const=True, default=True)
        parser.add_argument("--grad_clip_lars", type=str2bool, nargs='?',
                            const=True, default=True)
        parser.add_argument("--eta_lars", default=0.02, type=float)
        parser.add_argument("--exclude_bias_n_norm", type=str2bool, nargs='?',
                            const=True, default=True)

        # scheduler
        SUPPORTED_SCHEDULERS = [
            "reduce",
            "cosine",
            "warmup_cosine",
            "step",
            "exponential",
            "none",
        ]

        parser.add_argument("--scheduler", choices=SUPPORTED_SCHEDULERS, type=str, default="step")
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
        parser.add_argument("--min_lr", default=0.0, type=float)
        parser.add_argument("--warmup_start_lr", default=0.003, type=float)
        parser.add_argument("--warmup_epochs", default=10, type=int)

        # online knn eval
        parser.add_argument("--disable_knn_eval", default=True, action="store_false")
        parser.add_argument("--knn_k", default=20, type=int)

        return parent_parser

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "encoder", "params": self.encoder.parameters()},
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": 0,
            },
        ]

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        # collect learnable parameters
        idxs_no_scheduler = [
            i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)
        ]

        # select optimizer
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam)")

        # create optimizer
        optimizer = optimizer(
            self.learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )
        # optionally wrap with lars
        if self.lars:
            optimizer = LARSWrapper(
                optimizer,
                eta=self.eta_lars,
                clip=self.grad_clip_lars,
                exclude_bias_n_norm=self.exclude_bias_n_norm,
            )

        if self.scheduler == "none":
            return optimizer
        else:
            if self.scheduler == "warmup_cosine":
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=self.warmup_epochs,
                    max_epochs=self.max_epochs,
                    warmup_start_lr=self.warmup_start_lr,
                    eta_min=self.min_lr,
                )
            elif self.scheduler == "cosine":
                scheduler = CosineAnnealingLR(optimizer, self.max_epochs, eta_min=self.min_lr)
            elif self.scheduler == "step":
                scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
            else:
                raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

            if idxs_no_scheduler:
                partial_fn = partial(
                    static_lr,
                    get_lr=scheduler.get_lr,
                    param_group_indexes=idxs_no_scheduler,
                    lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
                )
                scheduler.get_lr = partial_fn

            return [optimizer], [scheduler]

    def forward(self, *args, **kwargs):
        """Dummy forward, calls base forward."""

        return self._base_forward(*args, **kwargs)

    def _base_forward(self, X: torch.Tensor) -> Dict:
        """Basic forward that allows children classes to override forward().

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """
        feats = self.encoder(X)
        logits = self.classifier(feats.detach())
        return {"logits": logits, "feats": feats}

    def _shared_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Forwards a batch of images X and computes the classification loss, the logits, the
        features, acc@1 and acc@5.

        Args:
            X (torch.Tensor): batch of images in tensor format
            targets (torch.Tensor): batch of labels for X

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5
        """

        out = self._base_forward(X)
        logits, feats = out["logits"], out["feats"]

        loss = self.loss_fn(logits, targets, **self.loss_args)
        results = self.metric_fn(logits, targets, **self.metric_args)

        return {
            "loss": loss,
            "logits": logits,
            "feats": feats,
            **results
        }

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It does all the shared operations, such as
        forwarding the crops, computing logits and computing statistics.

        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images
            batch_idx (int): index of the batch

        Returns:
            Dict[str, Any]: dict with the classification loss, features and logits
        """

        _, X, targets = batch
        targets = targets['labels'] if isinstance(targets, dict) else targets
        X = [X] if isinstance(X, torch.Tensor) else X

        # check that we received the desired number of crops
        # print(f"len(X) = {len(X)}")
        # print(f"self.n_crops = {self.n_crops}")
        # print(f"self.n_small_crops = {self.n_small_crops}")
        assert len(X) == self.n_crops + self.n_small_crops

        outs = [self._shared_step(x, targets) for x in X[: self.n_crops]]

        # collect data
        logits = [out["logits"] for out in outs]
        feats = [out["feats"] for out in outs]

        # loss and stats
        loss = sum(out["loss"] for out in outs) / self.n_crops
        metrics = {"train_class_loss": loss}

        for key in self.metric_keys:
            metrics.update({f"train_{key}": sum(out[key] for out in outs) / self.n_crops})

        if self.multicrop:
            feats.extend([self.encoder(x) for x in X[self.n_crops :]])

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        if not self.disable_knn_eval:
            self.knn(
                train_features=torch.cat(outs["feats"][:self.num_crops]).detach(),
                train_targets=targets.repeat(self.num_crops).detach(),
            )

        logit_cat = torch.cat(logits[:self.n_crops], dim=0)
        # print(f"logit_cat.shape = {logit_cat.shape}")
        scores = softmax(logit_cat)[:, 1]

        self.train_auroc.update(scores.detach(), targets.detach())

        out_dict = {
            "loss": loss,
            "feats": feats,
            "logits": logits,
        }
        return out_dict

    def training_epoch_end(self, outs: List[Dict[str, Any]]):
        auroc = self.train_auroc.compute()
        log = {"train_auroc": auroc}
        self.log_dict(log, sync_dist=True)
        self.train_auroc.reset()

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Validation step for pytorch lightning. It does all the shared operations, such as
        forwarding a batch of images, computing logits and computing metrics.

        Args:
            batch (List[torch.Tensor]):a batch of data in the format of [img_indexes, X, Y]
            batch_idx (int): index of the batch

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies
        """

        X, targets = batch
        targets = targets['labels'] if isinstance(targets, dict) else targets
        batch_size = targets.size(0)

        out = self._shared_step(X, targets)

        if not self.disable_knn_eval and not self.trainer.running_sanity_check:
            self.knn(test_features=out.pop("feats").detach(), test_targets=targets.detach())

        metrics = {
            "batch_size": batch_size,
            "val_loss": out["loss"],
        }
        for key in self.metric_keys:
            metrics.update({f"val_{key}": out[key]})

        scores = softmax(out["logits"])[:, 1]
        self.val_auroc.update(scores.detach(), targets.detach())

        return metrics

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """
        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        auroc = self.val_auroc.compute()
        self.val_auroc.reset()
        log = {"val_loss": val_loss, "val_auroc": auroc}
        for key in self.metric_keys:
            log.update({f"val_{key}": weighted_mean(outs, f"val_{key}", "batch_size")})

        if not self.disable_knn_eval and not self.trainer.running_sanity_check:
            val_knn_acc1, val_knn_acc5 = self.knn.compute()
            log.update({"val_knn_acc1": val_knn_acc1, "val_knn_acc5": val_knn_acc5})

        self.log_dict(log, sync_dist=True)


class BaseMomentumModel(BaseModel):
    def __init__(
        self,
        base_tau_momentum: float=0.99,
        final_tau_momentum: float=1.0,
        momentum_classifier: bool=False,
        **kwargs,
    ):
        """Base momentum model that implements all basic operations for all self-supervised methods
        that use a momentum encoder. It adds shared momentum arguments, adds basic learnable
        parameters, implements basic training and validation steps for the momentum encoder and
        classifier. Also implements momentum update using exponential moving average and cosine
        annealing of the weighting decrease coefficient.

        Args:
            base_tau_momentum (float): base value of the weighting decrease coefficient (should be
                in [0,1]).
            final_tau_momentum (float): final value of the weighting decrease coefficient (should be
                in [0,1]).
            momentum_classifier (bool): whether or not to train a classifier on top of the momentum
                encoder.
        """

        super().__init__(**kwargs)
        # momentum encoder
        self.momentum_encoder = self.base_model(**self.encoder_kwargs)
        if "resnet" in self.encoder_name:
            self.momentum_encoder.fc = nn.Identity()
            if self.cifar:
                self.momentum_encoder.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
                self.momentum_encoder.maxpool = nn.Identity()
        initialize_momentum_params(self.encoder, self.momentum_encoder)

        # momentum classifier
        if momentum_classifier:
            self.momentum_classifier: Any = nn.Linear(self.features_size, self.n_classes)
        else:
            self.momentum_classifier = None

        # momentum updater
        self.momentum_updater = MomentumUpdater(base_tau_momentum, final_tau_momentum)

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Adds momentum classifier parameters to the parameters of the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        momentum_learnable_parameters = []
        if self.momentum_classifier is not None:
            momentum_learnable_parameters.append(
                {
                    "name": "momentum_classifier",
                    "params": self.momentum_classifier.parameters(),
                    "lr": self.classifier_lr,
                    "weight_decay": 0,
                }
            )
        return super().learnable_params + momentum_learnable_parameters

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Defines base momentum pairs that will be updated using exponential moving average.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs (two element tuples).
        """

        return [(self.encoder, self.momentum_encoder)]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic momentum arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parent_parser = super(BaseMomentumModel, BaseMomentumModel).add_model_specific_args(
            parent_parser
        )
        parser = parent_parser.add_argument_group("base")

        # momentum settings
        parser.add_argument("--base_tau_momentum", default=0.99, type=float)
        parser.add_argument("--final_tau_momentum", default=1.0, type=float)
        parser.add_argument("--momentum_classifier", type=str2bool, nargs='?',
                            const=True)

        return parent_parser

    def on_train_start(self):
        """Resents the step counter at the beginning of training."""
        self.last_step = 0

    def _shared_step_momentum(self, X: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """Forwards a batch of images X in the momentum encoder and optionally computes the
        classification loss, the logits, the features, acc@1 and acc@5 for of momentum classifier.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict[str, Any]:
                a dict containing the classification loss, logits, features, acc@1 and
                acc@5 of the momentum encoder / classifier.
        """

        with torch.no_grad():
            feats = self.momentum_encoder(X)
        out = {"feats": feats}

        if self.momentum_classifier is not None:
            logits = self.momentum_classifier(feats)
            loss = self.loss_fn(logits, targets, **self.loss_args)
            results = self.metric_fn(logits, targets, **self.metric_args)
            out.update({"logits": logits, "loss": loss, **results})

        return out

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It performs all the shared operations for the
        momentum encoder and classifier, such as forwarding the crops in the momentum encoder
        and classifier, and computing statistics.

        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: a dict with the features of the momentum encoder and the classification
                loss and logits of the momentum classifier.
        """

        parent_outs = super().training_step(batch, batch_idx)

        _, X, targets = batch
        targets = targets['labels'] if isinstance(targets, dict) else targets
        X = [X] if isinstance(X, torch.Tensor) else X

        # remove small crops
        X = X[: self.n_crops]

        outs = [self._shared_step_momentum(x, targets) for x in X]

        # collect features
        parent_outs["feats_momentum"] = [out["feats"] for out in outs]

        if self.momentum_classifier is not None:
            # collect logits
            logits = [out["logits"] for out in outs]

            # momentum loss and stats
            loss = sum(out["loss"] for out in outs) / self.n_crops
            metrics = {"train_momentum_class_loss": loss}
            for key in self.metric_keys:
                metrics.update(
                    {f"train_momentum_{key}": sum(out[key] for out in outs) / self.n_crops}
                )

            self.log_dict(metrics, on_epoch=True, sync_dist=True)

            parent_outs["loss"] += loss
            parent_outs["logits_momentum"] = logits

        return parent_outs

    def on_train_batch_end(
        self, outputs: Dict[str, Any], batch: Sequence[Any], batch_idx: int, dataloader_idx: int
    ):
        """Performs the momentum update of momentum pairs using exponential moving average at the
        end of the current training step if an optimizer step was performed.

        Args:
            outputs (Dict[str, Any]): the outputs of the training step.
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
            batch_idx (int): index of the batch.
            dataloader_idx (int): index of the dataloader.
        """
        if self.trainer.global_step > self.last_step:
            # update momentum encoder and projector
            momentum_pairs = self.momentum_pairs
            for mp in momentum_pairs:
                self.momentum_updater.update(*mp)
            # log tau momentum
            self.log("tau", self.momentum_updater.cur_tau)
            # update tau
            self.momentum_updater.update_tau(
                cur_step=self.trainer.global_step * self.trainer.accumulate_grad_batches,
                max_steps=len(self.trainer.train_dataloader) * self.trainer.max_epochs,
            )
        self.last_step = self.trainer.global_step

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validation step for pytorch lightning. It performs all the shared operations for the
        momentum encoder and classifier, such as forwarding a batch of images in the momentum
        encoder and classifier and computing statistics.

        Args:
            batch (List[torch.Tensor]): a batch of data in the format of [X, Y].
            batch_idx (int): index of the batch.

        Returns:
            Tuple(Dict[str, Any], Dict[str, Any]): tuple of dicts containing the batch_size (used
                for averaging), the classification loss and accuracies for both the online and the
                momentum classifiers.
        """

        parent_metrics = super().validation_step(batch, batch_idx)

        X, targets = batch
        targets = targets['labels'] if isinstance(targets, dict) else targets
        batch_size = targets.size(0)

        out = self._shared_step_momentum(X, targets)

        metrics = None
        if self.momentum_classifier is not None:
            metrics = {
                "batch_size": batch_size,
                "momentum_val_loss": out["loss"]
            }
            for key in self.metric_keys:
                metrics.update({f"momentum_val_{key}": out[key]})

        return parent_metrics, metrics

    def validation_epoch_end(self, outs: Tuple[List[Dict[str, Any]]]):
        """Averages the losses and accuracies of the momentum encoder / classifier for all the
        validation batches. This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (Tuple[List[Dict[str, Any]]]):): list of outputs of the validation step for self
                and the parent.
        """

        parent_outs = [out[0] for out in outs]
        super().validation_epoch_end(parent_outs)

        if self.momentum_classifier is not None:
            momentum_outs = [out[1] for out in outs]
            val_loss = weighted_mean(momentum_outs, "momentum_val_loss", "batch_size")
            log = {"val_loss": val_loss}
            for key in self.metric_keys:
                stats = weighted_mean(momentum_outs, f"momentum_val_{key}", "batch_size")
                log.update({f"momentum_val_{key}": stats})
            self.log_dict(log, sync_dist=True)

