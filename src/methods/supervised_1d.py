from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Sequence, Tuple
import wandb
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.lars import LARSWrapper
from src.utils.metrics import accuracy_at_k, weighted_mean, AUROC, Get_target_and_preds_from_AUROC_object
from src.utils.blocks import str2bool
from src.utils.tricks import mixup_data, mixup_criterion, LabelSmoothingLoss
from src.methods.base import SUPPORTED_NETWORKS
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
)
# from pynvml import *
from torch.autograd import Variable

softmax = torch.nn.Softmax(dim=1)


class SupervisedModel_1D(pl.LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            n_classes: int,
            max_epochs: int,
            batch_size: int,
            optimizer: str,
            lars: bool,
            lr: float,
            weight_decay: float,
            exclude_bias_n_norm: bool,
            extra_optimizer_args: dict,
            scheduler: str,
            dataset: str,
            train_backbone: bool,
            mixup_alpha: float,
            label_smoothing: float,
            lr_decay_steps: Optional[Sequence[int]] = None,
            **kwargs,
    ):
        """Implements linear evaluation.

        Args:
            backbone (nn.Module): backbone architecture for feature extraction.
            n_classes (int): number of classes in the dataset.
            max_epochs (int): total number of epochs.
            batch_size (int): batch size.
            optimizer (str): optimizer to use.
            lars (bool): whether to use lars or not.
            lr (float): learning rate.
            weight_decay (float): weight decay.
            exclude_bias_n_norm (bool): whether to exclude bias and batch norm from weight decay
                and lars adaptation.
            extra_optimizer_args (dict): extra optimizer arguments.
            scheduler (str): learning rate scheduler.
            lr_decay_steps (Optional[Sequence[int]], optional): list of epochs where the learning
                rate will be decreased. Defaults to None.
        """

        super().__init__()

        self.backbone = backbone

        feat_in = self.backbone.inplanes if hasattr(self.backbone, 'inplanes') else self.backbone.num_features
        print(f"classifier feat_in = {feat_in}")
        self.classifier = nn.Linear(feat_in, n_classes)  # type: ignore

        # training related
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lars = lars
        self.lr = lr
        self.weight_decay = weight_decay
        self.exclude_bias_n_norm = exclude_bias_n_norm
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.lr_decay_steps = lr_decay_steps
        self.train_backbone = train_backbone
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        self.criterion = LabelSmoothingLoss(n_classes=2,
                                            smoothing=self.label_smoothing) if self.label_smoothing > 0 else nn.CrossEntropyLoss()

        # all the other parameters
        self.extra_args = kwargs

        # TODO: Implement this for supervised_2D
        self.pretrained_occlusion_model_dict = self.backbone.pretrained_occlusion_model_dict
        if self.pretrained_occlusion_model_dict is not None:
            self.flip_occlusion_model_grad(False)

        """ member for auroc calculation """
        self.train_auroc = AUROC(pos_label=1)
        self.val_auroc = AUROC(pos_label=1)
        self.test_auroc = AUROC(pos_label=1)
        self.buffer_train_auroc = -1
        self.max_val_auroc = -1
        self.corresponding_train_auroc = -1
        self.update_corresponding_train_auroc = True

        if "cifar" in dataset:
            self.backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            self.backbone.maxpool = nn.Identity()

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic linear arguments.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("supervised")

        # encoder args
        parser.add_argument("--encoder", choices=SUPPORTED_NETWORKS.keys(), type=str, default='resnet18')
        parser.add_argument("--zero_init_residual", type=str2bool, nargs='?', const=True)

        # general train
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--classifier_lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--num_workers", type=int, default=0)

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

        parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS,
                            type=str, default="sgd")
        parser.add_argument("--lars", type=str2bool, nargs='?',
                            const=True, default=True)
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
        parser.add_argument("--use_mask", type=str2bool, nargs='?', default=False)
        return parent_parser

    def flip_occlusion_model_grad(self, status: bool):
        """Sets requires_grad of inpainter (inference) model as True or False.

        Args:
            status (bool): determines whether requires_grad is True or False.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device = {device}")
        self.pretrained_occlusion_model_dict["mask_encoder"] = self.pretrained_occlusion_model_dict["mask_encoder"].to(
            device)
        self.pretrained_occlusion_model_dict["mask_head"] = self.pretrained_occlusion_model_dict["mask_head"].to(device)

        if self.pretrained_occlusion_model_dict is not None:
            for param in self.pretrained_occlusion_model_dict["mask_encoder"].parameters():
                param.requires_grad = status
            for param in self.pretrained_occlusion_model_dict["mask_head"].parameters():
                param.requires_grad = status

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """
        # print(f"[ecg] X.shape = {X.shape}")
        if self.pretrained_occlusion_model_dict is not None:
            X = self.pretrained_occlusion_model_dict["mask_head"](
                self.pretrained_occlusion_model_dict["mask_encoder"](X)
            )
            # print(f"[masks] X.shape = {X.shape}")
        feats = self.backbone(X)
        # print(f"feats.shape = {feats.shape}")
        logits = self.classifier(feats)
        # print(f"logits.shape = {logits.shape}")
        return {"logits": logits, "feats": feats}

    def configure_optimizers(self) -> Tuple[List, List]:
        """Configures the optimizer for the linear layer.

        Raises:
            ValueError: if the optimizer is not in (sgd, adam).
            ValueError: if the scheduler is not in not in (warmup_cosine, cosine, reduce, step,
                exponential).

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam)")
        model_parameters = list(self.backbone.parameters()) + list(self.classifier.parameters())
        optimizer = optimizer(
            model_parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        if self.lars:
            optimizer = LARSWrapper(optimizer, exclude_bias_n_norm=self.exclude_bias_n_norm)

        # select scheduler
        if self.scheduler == "none":
            return optimizer
        else:
            if self.scheduler == "warmup_cosine":
                scheduler = LinearWarmupCosineAnnealingLR(optimizer, 10, self.max_epochs)
            elif self.scheduler == "cosine":
                scheduler = CosineAnnealingLR(optimizer, self.max_epochs)
            elif self.scheduler == "reduce":
                scheduler = ReduceLROnPlateau(optimizer)
            elif self.scheduler == "step":
                scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.1)
            elif self.scheduler == "exponential":
                scheduler = ExponentialLR(optimizer, self.weight_decay)
            else:
                raise ValueError(
                    f"{self.scheduler} not in (warmup_cosine, cosine, reduce, step, exponential)"
                )

            return [optimizer], [scheduler]

    def shared_step(
            self, batch: Tuple, batch_idx: int
            , mode: str = "train") -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs operations that are shared between the training, validation and test steps.

        Args:
            batch (Tuple): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
                batch size, loss, accuracy @1 and accuracy @5.
        """
        X, targets = batch
        # print(f"batch, input dim = {X.shape}")
        if isinstance(X, list):
            X = torch.cat(X, dim=0)
            targets = torch.cat([targets, targets], dim=0)

        if mode == "train":
            use_cuda = torch.cuda.is_available()
            X, targets_a, targets_b, lam = mixup_data(X, targets, self.mixup_alpha, use_cuda)
            X, targets_a, targets_b = map(Variable, (X, targets_a, targets_b))
            out = self(X)["logits"]
            loss = mixup_criterion(self.criterion, out, targets_a, targets_b, lam)
        else:
            out = self(X)["logits"]
            loss = self.criterion(out, targets)

        # print(f"[{mode}] batch_size = {batch_size}, type(X) = {type(X)}, X.shape = {X.shape}, targets.shape = {targets.shape}")

        scores = softmax(out)[:, 1].detach()
        if mode in ["train"]:
            self.train_auroc.update(scores, targets.detach())
            # print(mode, len(self.train_auroc.preds))
        elif mode in ["val"]:
            self.val_auroc.update(scores, targets.detach())
        elif mode in ["test"]:
            self.test_auroc.update(scores, targets.detach())
        else:
            raise NotImplementedError("Unkown training mode.")

        results = accuracy_at_k(out, targets, top_k=(1,))
        batch_size = X.size(0)

        # return batch_size, loss, results['acc1'], results['acc5']
        return batch_size, loss, results['acc1'], results['acc1']
        # return batch_size, loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """

        # set encoder to eval mode
        if self.train_backbone:
            self.backbone.train()
        else:
            self.backbone.eval()

        _, loss, acc1, _ = self.shared_step(batch, batch_idx, mode="train")
        # _, loss = self.shared_step(batch, batch_idx, mode="train")

        log = {"train_loss": loss, "train_acc1": acc1, "train_acc5": float('nan')}
        self.log_dict(log, on_epoch=True, sync_dist=True)

        return loss

    def training_epoch_end(self, outs: List[Dict[str, Any]]):
        train_auroc = self.train_auroc.compute()
        # self.buffer_train_auroc = train_auroc
        if self.update_corresponding_train_auroc:
            self.corresponding_train_auroc = train_auroc

        self.log("train_auroc", train_auroc, on_epoch=True, sync_dist=True)
        self.log("corresponding_train_auroc", self.corresponding_train_auroc, on_epoch=True, sync_dist=True)
        # target, preds = Get_target_and_preds_from_AUROC_object(self.train_auroc)
        # title = "Train ROC"
        # if len(target) > 0 and len(preds) > 0:
        #     wandb.log({title: wandb.plot.roc_curve(target, preds, labels=["Sinus", "JET"], classes_to_plot=[1], title=title)})
        self.train_auroc.reset()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Performs the validation step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """
        batch_size, loss, acc1, _ = self.shared_step(batch, batch_idx, mode="val")

        results = {
            "batch_size": batch_size,
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": float('nan'),
        }

        return results

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """
        val_auroc = self.val_auroc.compute()
        self.log("val_auroc", val_auroc, on_epoch=True, sync_dist=True)
        if val_auroc > self.max_val_auroc or self.corresponding_train_auroc < 0:
            self.max_val_auroc = val_auroc
            self.update_corresponding_train_auroc = True
        else:
            self.update_corresponding_train_auroc = False

        self.log("max_val_auroc", self.max_val_auroc, on_epoch=True, sync_dist=True)
        target, preds = Get_target_and_preds_from_AUROC_object(self.val_auroc)
        title = "Validation ROC"
        if len(target) > 0 and len(preds) > 0:
            try:
                wandb.log({title: wandb.plot.roc_curve(target, preds, labels=["Sinus", "JET"], classes_to_plot=[1], title=title)})
            except:
                print(f"target: {target}")
                print(f"preds: {preds}")
        self.val_auroc.reset()

        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        # val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")
        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": float('nan')}
        self.log_dict(log, sync_dist=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Performs the test step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """

        batch_size, loss, acc1, _ = self.shared_step(batch, batch_idx, mode="test")

        results = {
            "batch_size": batch_size,
            "test_loss": loss,
            "test_acc1": acc1,
            "test_acc5": float('nan'),
        }
        return results

    def test_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the test batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """
        test_auroc = self.test_auroc.compute()
        self.log("test_auroc", test_auroc, on_epoch=True, sync_dist=True)
        target, preds = Get_target_and_preds_from_AUROC_object(self.test_auroc)
        title = "Test ROC"
        if len(target) > 0 and len(preds) > 0:
            wandb.log({title: wandb.plot.roc_curve(target, preds, labels=["Sinus", "JET"], classes_to_plot=[1], title=title)})
        self.test_auroc.reset()

        test_loss = weighted_mean(outs, "test_loss", "batch_size")
        test_acc1 = weighted_mean(outs, "test_acc1", "batch_size")
        # test_acc5 = weighted_mean(outs, "test_acc5", "batch_size")
        log = {"test_loss": test_loss, "test_acc1": test_acc1, "test_acc5": float('nan')}
        self.log_dict(log, sync_dist=True)
