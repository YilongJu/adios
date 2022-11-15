import argparse
from typing import Any, Dict, List, Tuple, Callable, Sequence
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.lars import LARSWrapper
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.losses.simclr import simclr_loss_func
from src.methods.base_adios import BaseADIOSModel
from src.utils.unet import UNet
from src.utils.unet_1d import UNet_1D, UNet_1D_separate
from src.utils.metrics import Entropy


class DISENTANGLED_ADIOS_1D(pl.LightningModule):
    def __init__(
            self,
            output_dim: int,
            proj_hidden_dim: int,
            temperature: float,
            mask_lr: float,
            alpha_sparsity: float,
            alpha_entropy: float,
            N: int,
            mask_fbase: int,
            unet_norm: str,
            **kwargs
    ):
        """Implements a disentangled contrastive learning framework inspired by ADIOS.
        In total, 3 networks are used:
            1) Domain-info extraction network ($f_D$),
            2) Class-info extraction network ($f_C$),
            3) Signal modification network ($f_S$).
            - They may all share weights for the encoder part.
        See OneNote 2021 Spring - ECG for forward pass procedure.

        Args:
            output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
            mask_lr (float): learning rate of masking model.
            alpha_sparsity (float): mask penalty term.
            alpha_entropy (float): mask penalty term 2
            N (int): number of masks to use.
            mask_fbase (int): base channels of masking model.
            unet_norm (str): normalisation function, choose from
                    - "no": no normalisation
                    - "in": instance noramlisation,
                    - "gn": group normalisation
        """

        super().__init__()

        self.temperature = temperature
        self.mask_lr = mask_lr
        self.mask_lr = self.mask_lr * self.accumulate_grad_batches
        self.alpha1 = alpha_sparsity
        self.alpha2 = alpha_entropy
        self.N = N
        self.entropy = Entropy()

        # # simclr projector
        # self.domain_info_network = nn.Sequential(
        #     nn.Linear(self.features_size, proj_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(proj_hidden_dim, output_dim),
        # )
        # # simclr projector
        # self.class_info_network = nn.Sequential(
        #     nn.Linear(self.features_size, proj_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(proj_hidden_dim, output_dim),
        # )
        # Activates manual optimization
        self.automatic_optimization = False

        # Signal Modification Network f_S
        self.signal_modification_network_head = UNet_1D_separate(
            num_blocks=int(np.log2(self.img_size) - 1),
            img_size=self.img_size,
            filter_start=mask_fbase,
            in_chnls=1,
            out_chnls=-1,
            norm=unet_norm).to(torch.float)

        """ From SimSiam (github.com/facebookresearch/simsiam) """
        encoder_output_size = 256  # From signal_modification_network
        # Domain-info Extraction Network f_D
        self.domain_info_network_head = nn.Sequential(
            nn.Linear(encoder_output_size, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, output_dim),
        )
        # Class-info Extraction Network f_C
        self.class_info_network_head = nn.Sequential(
            nn.Linear(encoder_output_size, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, output_dim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(DISENTANGLED_ADIOS_1D, DISENTANGLED_ADIOS_1D).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simclr")
        # simclr args
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        parser.add_argument("--temperature", type=float, default=0.1)
        # adios args
        parser.add_argument("--mask_lr", type=float, default=0.1)
        parser.add_argument("--alpha_sparsity", type=float, default=1.)
        parser.add_argument("--alpha_entropy", type=float, default=0.)
        parser.add_argument("--N", type=int, default=6)
        parser.add_argument("--mask_fbase", type=int, default=32)
        parser.add_argument("--unet_norm", type=str, default='gn',
                            choices=['gn', 'in', 'no'])
        return parent_parser

    @property
    def learnable_params(self) -> Dict[str, Any]:
        """Adds projector and masking model parameters to the parent's learnable parameters.

        Returns:
            Dict[str, Any]: dictionary of learnable parameters.
        """
        # super().learnable_params
        # inpainter_learnable_params = [
        #     {"params": self.projector.parameters()}
        # ]
        # mask_learnable_params = [
        #     {
        #         "name": "mask_encoder",
        #         "params": self.mask_encoder.parameters(),
        #         "lr": self.mask_lr
        #     },
        #     {
        #         "name": "mask_head",
        #         "params": self.mask_head.parameters(),
        #         "lr": self.mask_lr
        #     }
        # ]
        network_learnable_params = [
            {
                "name": "encoder",
                "params": self.encoder.parameters(),
                "lr": self.lr
            },
            {
                "name": "domain_info_network_head",
                "params": self.domain_info_network_head.parameters(),
                "lr": self.lr
            },
            {
                "name": "class_info_network_head",
                "params": self.class_info_network_head.parameters(),
                "lr": self.lr
            }
        ]
        return {"all": network_learnable_params}

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        # select optimizer
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam)")

        # create optimizer
        optimizer = [optimizer(
            self.learnable_params["all"],
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )]

        # optionally wrap with lars
        if self.lars:
            optimizer = [LARSWrapper(
                opt,
                eta=self.eta_lars,
                clip=self.grad_clip_lars,
                exclude_bias_n_norm=self.exclude_bias_n_norm,
            ) for opt in optimizer]

        if self.scheduler == "none":
            return optimizer  # todo: might need some touch up due to new optimiser structure
        else:
            if self.scheduler == "warmup_cosine":
                scheduler = [
                    LinearWarmupCosineAnnealingLR(
                        optimizer[0],
                        warmup_epochs=self.warmup_epochs,
                        max_epochs=self.max_epochs,
                        warmup_start_lr=self.warmup_start_lr,
                        eta_min=self.min_lr,
                    )]
            elif self.scheduler == "cosine":
                scheduler = [CosineAnnealingLR(optimizer[0], self.max_epochs, eta_min=self.min_lr)]
            elif self.scheduler == "step":
                scheduler = [MultiStepLR(opt, self.lr_decay_steps) for opt in optimizer]
            else:
                raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

            return optimizer, scheduler

    # def forward(self, X: torch.tensor, *args, **kwargs) -> Dict[str, Any]:
    #     """Performs the forward pass of the encoder, the projector and the predictor.
    #
    #     Args:
    #         X (torch.Tensor): a batch of images in the tensor format.
    #
    #     Returns:
    #         Dict[str, Any]:
    #             a dict containing the outputs of the parent
    #             and the projected and predicted features.
    #     """
    #
    #     out = super().forward(X, *args, **kwargs) # base.forward (ResNet1D.forward): out = {"logits": logits, "feats": feats}
    #     z = self.projector(out["feats"])
    #     return {**out, "z": z}

    def flip_network_requires_grad(self, status: bool):
        """Sets requires_grad of inpainter (inference) model as True or False.

        Args:
            status (bool): determines whether requires_grad is True or False.
        """
        for param in self.encoder.parameters():
            param.requires_grad = status
        for param in self.domain_info_network_head.parameters():
            param.requires_grad = status
        for param in self.class_info_network_head.parameters():
            param.requires_grad = status

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """
        # print(f"[ecg] X.shape = {X.shape}")
        X_masked, latent_representation = self.encoder(X)
        embedding_domain = self.domain_info_network_head(latent_representation)
        embedding_class = self.class_info_network_head(latent_representation)
        # print(f"embedding_domain.shape = {embedding_domain.shape}")
        # print(f"embedding_class.shape = {embedding_class.shape}")
        return {"X_masked": X_masked, "latent_representation": latent_representation,
                "embedding_domain": embedding_domain,
                "embedding_class": embedding_class}

    def shared_step(self, batch: Tuple, mode: str = "train"):
        """Performs operations that are shared between the training and validation steps.

        Args:
            batch (Tuple): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
                batch size, loss, accuracy @1 and accuracy @5.
        """
        X_aug_a, X_aug_b, X_br_a, X_br_b, y = batch  # X_aug_a, X_aug_b: augmented ECGs, X_br: baseline removed ECG, y: label
        # print(f"[ecg] X_aug_a.shape = {X_aug_a.shape}")
        X_masked_a, r_aug_a_latent = self.encoder(X_aug_a)
        X_masked_b, r_aug_b_latent = self.encoder(X_aug_b)
        z_aug_a_domain = self.domain_info_network_head(r_aug_a_latent)
        z_aug_b_domain = self.domain_info_network_head(r_aug_b_latent)
        z_aug_a_class = self.class_info_network_head(r_aug_a_latent)
        z_aug_b_class = self.class_info_network_head(r_aug_b_latent)
        _, r_masked_a_latent = self.encoder(X_masked_a)
        _, r_masked_b_latent = self.encoder(X_masked_b)
        z_masked_a_class = self.class_info_network_head(r_masked_a_latent)
        z_masked_b_class = self.class_info_network_head(r_masked_b_latent)
        X_bg_a = X_aug_a - X_masked_a
        X_bg_b = X_aug_b - X_masked_b
        _, r_bg_a_latent = self.encoder(X_bg_a)
        _, r_bg_b_latent = self.encoder(X_bg_b)
        z_bg_a_domain = self.domain_info_network_head(r_bg_a_latent)
        z_bg_b_domain = self.domain_info_network_head(r_bg_b_latent)
        return_list = [
            z_aug_a_domain, z_aug_b_domain, z_aug_a_class, z_aug_b_class,
            z_masked_a_class, z_masked_b_class, z_bg_a_domain, z_bg_b_domain,
            X_br_a, X_br_b, X_bg_a, X_bg_b
        ]

        return return_list

    def training_step(self, batch: Sequence[Any], batch_idx: int):
        """Training step for SimCLR ADIOS.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
            batch_idx (int): index of the batch.
        """

        """ Loss functions """
        # z_aug_a_domain, z_aug_b_domain, z_aug_a_class, z_aug_b_class, \
        #     z_masked_a_class, z_masked_b_class, z_bg_a_domain, z_bg_b_domain, \
        #     X_br_a, X_br_b, X_bg_a, X_bg_b = self.shared_step(batch, mode="train")

        X_aug_a, X_aug_b, X_br_a, X_br_b, y = batch  # X_aug_a, X_aug_b: augmented ECGs, X_br: baseline removed ECG, y: label
        # print(f"[ecg] X_aug_a.shape = {X_aug_a.shape}")
        X_masked_a, r_aug_a_latent = self.encoder(X_aug_a)
        X_masked_b, r_aug_b_latent = self.encoder(X_aug_b)
        z_aug_a_domain = self.domain_info_network_head(r_aug_a_latent)
        z_aug_b_domain = self.domain_info_network_head(r_aug_b_latent)
        z_aug_a_class = self.class_info_network_head(r_aug_a_latent)
        z_aug_b_class = self.class_info_network_head(r_aug_b_latent)
        _, r_masked_a_latent = self.encoder(X_masked_a)
        _, r_masked_b_latent = self.encoder(X_masked_b)
        z_masked_a_class = self.class_info_network_head(r_masked_a_latent)
        z_masked_b_class = self.class_info_network_head(r_masked_b_latent)
        X_bg_a = X_aug_a - X_masked_a
        X_bg_b = X_aug_b - X_masked_b
        _, r_bg_a_latent = self.encoder(X_bg_a)
        _, r_bg_b_latent = self.encoder(X_bg_b)
        z_bg_a_domain = self.domain_info_network_head(r_bg_a_latent)
        z_bg_b_domain = self.domain_info_network_head(r_bg_b_latent)
        # return_list = [
        #     z_aug_a_domain, z_aug_b_domain, z_aug_a_class, z_aug_b_class,
        #     z_masked_a_class, z_masked_b_class, z_bg_a_domain, z_bg_b_domain,
        #     X_br_a, X_br_b, X_bg_a, X_bg_b
        # ]
        loss_type = "mean_distance_to_center"
        alpha_class = 1.
        if loss_type == "mean_distance_to_center":
            def Compute_mean_dist(z_tensor, ord=2):
                z_mean = torch.mean(z_tensor, dim=2)
                z_centered = z_tensor - z_mean.unsqueeze(2)
                z_dist = torch.sqrt(torch.sum(torch.abs(z_centered) ** ord, dim=1))
                z_dist_mean = torch.mean(z_dist, dim=1)
                return z_dist_mean

            z_domain_all = torch.cat([z_aug_a_domain.unsqueeze(2), z_aug_b_domain.unsqueeze(2),
                                      z_bg_a_domain.unsqueeze(2), z_bg_b_domain.unsqueeze(2)], dim=2)
            z_domain_dist_mean = Compute_mean_dist(z_domain_all)
            z_class_all = torch.cat([z_aug_a_class.unsqueeze(2), z_aug_b_class.unsqueeze(2),
                                     z_masked_a_class.unsqueeze(2), z_masked_b_class.unsqueeze(2)], dim=2)
            z_class_dist_mean = Compute_mean_dist(z_class_all)
            loss = torch.mean(z_domain_dist_mean) + alpha_class * torch.mean(z_class_dist_mean)
        else:
            raise ValueError("loss_type not implemented.")

        return loss


    def linear_forward(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Forward function for linear classifier (backbone model is detached).

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
            batch_idx (int): index of the batch.
        """
        indexes, *_, target = batch
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        return class_loss

    def inpaint_forward(self, batch: Sequence[Any]) -> torch.Tensor:
        """Forward function for inpainter (inference model).

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
        """
        # TODO: Add transformation for ECG """
        # indexes, [x_orig, x_transformed], target = batch
        # Untransformed pair for now (20220629)
        indexes, x_orig, target = batch
        x_transformed = x_orig
        enc_feat = self.mask_encoder(x_transformed)
        masks = self.mask_head(enc_feat)

        similarities = []
        for k, mask in enumerate(torch.chunk(masks, self.N, dim=1)):
            mask = mask.detach()
            feats1, feats2 = self.encoder(x_orig), self.encoder(x_transformed * (1 - mask))
            z1 = self.projector(feats1)
            z2 = self.projector(feats2)

            # compute similarity between mask and no mask
            loss_func_kwargs = {"temperature": self.temperature, "pos_only": False}
            similarities.append(simclr_loss_func(z1, z2, **loss_func_kwargs))

        similarity = torch.stack(similarities).sum()
        return similarity

    def mask_forward(self, batch: Sequence[Any]) -> torch.Tensor:
        """Forward function for masking model (occlusion model).

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
        """
        # TODO: Add transformation for ECG """
        # indexes, [x_orig, x_transformed], target = batch
        # Untransformed pair for now (20220630)
        indexes, x_orig, target = batch
        x_transformed = x_orig
        enc_feat = self.mask_encoder(x_transformed)
        masks = self.mask_head(enc_feat)

        similarities, summed_mask = [], []
        mask_var_list = []
        for k, mask in enumerate(torch.chunk(masks, self.N, dim=1)):
            feats1, feats2 = self.encoder(x_orig), self.encoder(x_transformed * (1 - mask))
            z1 = self.projector(feats1)
            z2 = self.projector(feats2)

            # compute similarity between mask and no mask
            loss_func_kwargs = {"temperature": self.temperature, "pos_only": False}
            loss = simclr_loss_func(z1, z2, **loss_func_kwargs)

            # compute mask penalty
            mask_var_list.append(torch.var(mask, dim=2).squeeze())
            mask_n_ele = torch.prod(torch.tensor(mask.shape)) / mask.shape[0]
            # print(f"mask.shape = {mask.shape}") # B x 1 x 300
            # print(f"mask = {mask}")
            # print(f"self.img_size = {self.img_size}") # B x 1 x 300
            # print(f"mask_n_ele = {mask_n_ele}")
            sm = mask.sum([-1, -2]) / mask_n_ele  # (B,)
            summed_mask.append(sm)
            loss -= self.alpha1 * (1 / (torch.sin(sm * np.pi) + 1e-10)).mean(0).sum(0)

            similarities.append(loss)

        similarity = torch.stack(similarities).sum()
        # print(f"summed_mask[-1].shape = {summed_mask[-1].shape}")
        # print(f"summed_mask[-1] = {summed_mask[-1]}")
        # print(f"summed_mask = {summed_mask}")
        all_summed_masks = torch.stack(summed_mask, dim=1)
        # print(f"all_summed_masks.shape = {all_summed_masks.shape}")
        # print(f"all_summed_masks = {all_summed_masks}")
        all_masks_vars = torch.cat(mask_var_list)

        similarity += self.alpha2 * self.entropy(all_summed_masks)
        minval, _ = torch.stack(summed_mask).min(dim=0)
        maxval, _ = torch.stack(summed_mask).max(dim=0)
        self.log_dict({"train_mask_loss": similarity,
                       "mask_summed_min": minval.mean(),
                       "mask_summed_max": maxval.mean(),
                       "mask_var_mean": all_masks_vars.mean()},
                      on_epoch=True, sync_dist=True)

        return similarity
