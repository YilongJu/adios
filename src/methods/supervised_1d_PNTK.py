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
from functorch import make_functional_with_buffers, vmap, grad, jacrev

# from pynvml import *
from torch.autograd import Variable
softmax = torch.nn.Softmax(dim=1)

def Flatten_per_sample_grads(per_sample_grads):
    per_sample_grads_flattened_list = []
    for per_sample_grad in per_sample_grads:
        ndat_dim = per_sample_grad.shape[0]
        per_sample_grad_flattened = per_sample_grad.view(ndat_dim, -1)
        per_sample_grads_flattened_list.append(per_sample_grad_flattened)

    per_sample_grads_flattened = torch.cat(per_sample_grads_flattened_list, dim=1)
    return per_sample_grads_flattened


def Flatten_per_sample_jacs(per_sample_jacs):
    per_sample_jacs_flattened_list = []
    for per_sample_jac in per_sample_jacs:
        ndat_dim = per_sample_jac.shape[0]
        nout_dim = per_sample_jac.shape[1]
        per_sample_grad_flattened = per_sample_jac.view(ndat_dim, nout_dim, -1).transpose(1, 2)
        per_sample_jacs_flattened_list.append(per_sample_grad_flattened)

    per_sample_jacs_flattened = torch.cat(per_sample_jacs_flattened_list, dim=1)
    return per_sample_jacs_flattened

class SupervisedModel_1D_PNTK(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
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
        Dout: Optional[int] = 2,
        subsample_factor: Optional[int] = 1,
        **kwargs,
    ):
        """Implements linear evaluation.

        Args:
            net (nn.Module): full architecture for feature extraction, which produces raw logits.
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

        self.net = net

        feat_in = self.net.inplanes if hasattr(self.net, 'inplanes') else self.net.num_features
        # self.classifier = nn.Linear(feat_in, n_classes)  # type: ignore

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

        # # TODO: Implement this for supervised_2D
        # self.pretrained_occlusion_model_dict = self.net.pretrained_occlusion_model_dict
        # if self.pretrained_occlusion_model_dict is not None:
        #     self.flip_occlusion_model_grad(False)

        """ member for auroc calculation """
        self.train_auroc = AUROC(pos_label=1)
        self.val_auroc = AUROC(pos_label=1)
        self.test_auroc = AUROC(pos_label=1)
        self.buffer_train_auroc = -1
        self.max_val_auroc = -1
        self.corresponding_train_auroc = -1
        self.update_corresponding_train_auroc = True

        """ For PNTK calculation """
        self.automatic_optimization = False # Controls pytorch-lightning's automatic optimization
        self.subsample_factor = subsample_factor
        self.lr_sub = self.lr / self.subsample_factor

        self.Dout = Dout
        self.train_batches = None
        print("Dout = ", self.Dout)
        print("train_batches = ", self.train_batches)
        self.PNTK_logging_dict = {
            "meanlogiterror": [],
            "mismatches": [],
            "meanlogitchange": [],
            "truemeanlogitchange": [],
            "PNTK": None,
            "predict_yinit": None,
            "test_X_batch_0": None,
            "last_out": None
        }
        fmodel, _, buffers = make_functional_with_buffers(net)
        ft_compute_grad = grad(self.compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        ft_compute_jac = jacrev(self.compute_output_stateless_model)
        ft_compute_sample_jac = vmap(ft_compute_jac, in_dims=(None, None, 0))

        self.PNTK_functorch_dict = {
            "fmodel": fmodel,
            "buffers": buffers,
            "ft_compute_grad": ft_compute_grad,
            "ft_compute_sample_grad": ft_compute_sample_grad,
            "ft_compute_jac": ft_compute_jac,
            "ft_compute_sample_jac": ft_compute_sample_jac,
        }

        # self.PNTK_functorch_dict = {
        #     "fmodel": None,
        #     "buffers": None,
        #     "ft_compute_grad": None,
        #     "ft_compute_sample_grad": None,
        #     "ft_compute_jac": None,
        #     "ft_compute_sample_jac": None,
        # }

        if "cifar" in dataset:
            self.net.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            self.net.maxpool = nn.Identity()

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
        """Performs forward pass of the net.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """
        # print(f"[ecg] X.shape = {X.shape}")
        # if self.pretrained_occlusion_model_dict is not None:
        #     X = self.pretrained_occlusion_model_dict["mask_head"](
        #         self.pretrained_occlusion_model_dict["mask_encoder"](X)
        #     )
        #     # print(f"[masks] X.shape = {X.shape}")
        # feats = self.net(X)
        # print(f"feats.shape = {feats.shape}")
        # logits = self.classifier(feats)

        logits = self.net(X)
        # print(f"logits.shape = {logits.shape}")
        return {"logits": logits, "feats": None}

    def compute_loss_stateless_model(self, params, buffers, sample, target):
        """ for functorch """
        inputs = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        outputs = self.PNTK_functorch_dict["fmodel"](params, buffers, inputs)
        loss = self.criterion(outputs, targets)
        assert self.mixup_alpha == 0.0 # only work for 0 mixup
        return loss

    def compute_output_stateless_model(self, params, buffers, sample):
        inputs = sample.unsqueeze(0)
        outputs = self.PNTK_functorch_dict["fmodel"](params, buffers, inputs)
        output = outputs.squeeze(0)
        return output


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
        # model_parameters = list(self.net.parameters()) + list(self.classifier.parameters())
        model_parameters = list(self.net.parameters())
        optimizer = optimizer(
            model_parameters,
            lr=self.lr_sub,
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
        # print(f"self.current_epoch = {self.current_epoch}, self.global_step = {self.global_step}")

        X, targets = batch
        # print(f"batch, input dim = {X.shape}")
        if isinstance(X, list):
            X = torch.cat(X, dim=0)
            targets = torch.cat([targets, targets], dim=0)

        # print(f"X.shape = {X.shape}") # X.shape = torch.Size([512, 1, 300])
        # if mode == "train":
        #     use_cuda = torch.cuda.is_available()
        #     X, targets_a, targets_b, lam = mixup_data(X, targets, self.mixup_alpha, use_cuda)
        #     X, targets_a, targets_b = map(Variable, (X, targets_a, targets_b))
        #     out = self(X)["logits"]
        #     loss = mixup_criterion(self.criterion, out, targets_a, targets_b, lam)
        # else:
        #     out = self(X)["logits"]
        #     loss = self.criterion(out, targets)

        out = self(X)["logits"]
        loss = self.criterion(out, targets)
        # print(f"loss: {loss}")

        test_X_batch_0 = self.PNTK_logging_dict["test_X_batch_0"]
        if self.current_epoch == 0 and self.global_step == 0:
            # self.train_batches = self.trainer.num_training_batches
            # print(f"self.train_batches: {self.train_batches}")
            #
            # self.PNTK_logging_dict["PNTK"] = torch.zeros(self.batch_size * self.train_batches, self.batch_size, self.Dout)

            self.PNTK_logging_dict["predict_yinit"] = self(test_X_batch_0)["logits"].detach()

        self.PNTK_logging_dict["last_out"] = self(test_X_batch_0)["logits"].detach()
        do_PNTK = True
        # do_PNTK = False
        if do_PNTK:
            with torch.no_grad():
                # Yilong-style
                params = list(self.parameters())  # update params here
                # print(f"N params = {len(params)}")
                # print(f"params.shape = {[ele.shape for ele in params]}")
                ft_per_sample_grads = self.PNTK_functorch_dict["ft_compute_sample_grad"](
                    params, self.PNTK_functorch_dict["buffers"], X, targets) # grab per-training sample grads
                # print(f"ft_per_sample_grads = {ft_per_sample_grads}")
                # print(f"ft_per_sample_grads.shape = {[ele.shape for ele in ft_per_sample_grads]}")

                NTKtrain_torch = Flatten_per_sample_grads(ft_per_sample_grads)  # flatten them into 1D list for later dot producting
                # print(f"NTKtrain_torch.shape = {NTKtrain_torch.shape}")
                # print(f"NTKtrain_torch:\n{NTKtrain_torch}")

        if mode in ["train"]:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            for ss in range(self.subsample_factor):
                params = list(self.parameters())  # update params here, seems reasonable enough

                ft_per_sample_jacs = self.PNTK_functorch_dict["ft_compute_sample_jac"](params, self.PNTK_functorch_dict["buffers"], test_X_batch_0)  # Generate per-testing sample jacobians?
                NTKtest_torch = Flatten_per_sample_jacs(
                    ft_per_sample_jacs)  # flatten them into 1D list for later dot producting

                opt.step()
                batch_size = self.batch_size
                print(f"NTKtrain_torch.shape = {NTKtrain_torch.shape}")
                # TODO: Logit dimension = 2 not 16. fix layer that connects to PNTK calculation
                print(f"NTKtest_torch.shape = {NTKtest_torch.shape}")



                self.PNTK_logging_dict["PNTK"][batch_idx * batch_size:(batch_idx + 1) * batch_size, :, :] += -1 * self.lr_sub * torch.einsum('ik, jkl->ijl', NTKtrain_torch, NTKtest_torch) / batch_size
                # print(f"[{mode}] batch_size = {batch_size}, type(X) = {type(X)}, X.shape = {X.shape}, targets.shape = {targets.shape}")

            del NTKtrain_torch, NTKtest_torch

            new_out = self(test_X_batch_0)["logits"].detach()
            mean_logit_error = torch.mean(torch.abs(torch.sum(self.PNTK_logging_dict["PNTK"], 0) + self.PNTK_logging_dict["predict_yinit"] - new_out)).detach()
            mismatches = torch.sum(torch.argmax(new_out, 1) != torch.argmax(torch.sum(self.PNTK_logging_dict["PNTK"], 0) + self.PNTK_logging_dict["predict_yinit"], 1)).detach()
            true_mean_logit_change = torch.mean(torch.abs(new_out - self.PNTK_logging_dict["last_out"])).detach()


            log = {"mean_logit_error": mean_logit_error, "mismatches": mismatches, "true_mean_logit_change": true_mean_logit_change}
            self.log_dict(log, on_epoch=True, sync_dist=True)


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
            self.net.train()
        else:
            self.net.eval()

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
