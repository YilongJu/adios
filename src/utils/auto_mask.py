import os
from argparse import ArgumentParser, Namespace
from typing import Optional
from torchvision.utils import save_image

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from src.utils.ECG_data_plotting import Convert_batch_of_time_series_to_batch_of_img_torch_array
import wandb

from typing import Union, Optional, List, Tuple, Text, BinaryIO
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont, ImageColor
import pathlib
@torch.no_grad()
def save_image_with_return_img_list(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs
):
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # print(f"ndarr.shape: {ndarr.shape}")
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
    return [im]


class AutoMASK(Callback):
    def __init__(
        self,
        args: Namespace,
        logdir: str,
        frequency: int = 1,
        keep_previous: bool = False,
        color_palette: str = "hls",
    ):
        """UMAP callback that automatically runs UMAP on the validation dataset and uploads the
        figure to wandb.

        Args:
            args (Namespace): namespace object containing at least an attribute name.
            logdir (str, optional): base directory to store checkpoints.
                Defaults to "auto_umap".
            frequency (int, optional): number of epochs between each UMAP. Defaults to 1.
            color_palette (str, optional): color scheme for the classes. Defaults to "hls".
            keep_previous (bool, optional): whether to keep previous plots or not.
                Defaults to False.
        """

        super().__init__()

        self.args = args
        self.logdir = logdir
        self.frequency = frequency
        self.color_palette = color_palette
        self.keep_previous = keep_previous

        self.colors = [(255, 255, 255), (0, 0, 255), (0, 255, 255), (255, 0, 255),
                  (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        self.colors.extend([(c[0] // 2, c[1] // 2, c[2] // 2) for c in self.colors])
        self.colors.extend([(c[0] // 4, c[1] // 4, c[2] // 4) for c in self.colors])
        self.colors = [torch.tensor(c) for c in self.colors]

    @staticmethod
    def add_auto_mask_args(parent_parser: ArgumentParser):
        """Adds user-required arguments to a parser.

        Args:
            parent_parser (ArgumentParser): parser to add new args to.
        """

        parser = parent_parser.add_argument_group("auto_mask")
        parser.add_argument("--auto_mask_dir", default="auto_mask", type=str)
        parser.add_argument("--auto_mask_frequency", default=1, type=int)
        parser.add_argument("--img_per_row", default=8, type=int)
        parser.add_argument("--auto_mask_n_batches", default=3000, type=int)
        parser.add_argument("--mask_plot_type", default="soft", type=str)
        return parent_parser

    def initial_setup(self, trainer: pl.Trainer):
        """Creates the directories and does the initial setup needed.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if trainer.logger is None:
            version = None
        else:
            version = str(trainer.logger.version)
        if version is not None:
            self.path = os.path.join(self.logdir, trainer.logger.name, version)
        else:
            self.path = self.logdir

        self.umap_placeholder = "ep={}-n={}.png"
        self.last_ckpt: Optional[str] = None

        # create logging dirs
        if trainer.is_global_zero:
            os.makedirs(self.path, exist_ok=True)

    def on_train_start(self, trainer: pl.Trainer, _):
        """Performs initial setup on training start.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        self.initial_setup(trainer)

    def plot(self, trainer: pl.Trainer, module: pl.LightningModule):
        """Produces a UMAP visualization by forwarding all data of the
        first validation dataloader through the module.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
            module (pl.LightningModule): current module object.
        """

        """ Set seeds for consistent plotting """
        if self.args.ptl_accelerator in ["ddp"]:
            for v in trainer.val_dataloaders:
                v.sampler.shuffle = True
                v.sampler.set_epoch(0)
                print(f"v.sampler.set_epoch(0)")

        if "none" not in self.path:
            device = module.device
            # set module to eval model and collect all feature representations
            module.eval()
            with torch.no_grad():
                for n, (x, y) in enumerate(trainer.val_dataloaders[0]):
                    if n == self.args.auto_mask_n_batches:
                        break

                    x = x.to(device, non_blocking=True)[:self.args.img_per_row] # B * C * W * H / B * 1 * W
                    y = y.to(device, non_blocking=True)[:self.args.img_per_row].int() # B
                    feats = module.mask_encoder(x)
                    soft_masks = module.mask_head(feats)

                    if self.args.mask_plot_type == "soft":
                        masks_plot = soft_masks.cpu()
                    elif self.args.mask_plot_type == "hard":
                        a = soft_masks.argmax(dim=1).cpu()
                        hard_masks = torch.zeros(soft_masks.shape).scatter(1, a.unsqueeze(1), 1.0)
                        masks_plot = hard_masks

                    save_tensor = []
                    if len(x.shape) == 4:
                        input_img = x.cpu()
                        save_tensor.append(input_img)
                    elif len(x.shape) == 3: # Time series
                        """
                        References: 
                        - https://www.tutorialspoint.com/how-to-convert-matplotlib-figure-to-pil-image-object
                        - https://stackoverflow.com/questions/384759/how-to-convert-a-pil-image-into-a-numpy-array
                        """
                        pass
                    else:
                        raise NotImplementedError("Unknown data shape.")

                    for i, mask in enumerate(torch.chunk(masks_plot, self.args.N, dim=1)): # Split B * C mask tensor into separate B * 1 masks
                        if len(mask.shape) == 4:
                            save_tensor.extend([mask.repeat(1,3,1,1), input_img * (1 - mask)])
                        elif len(mask.shape) == 3:
                            # 'mask' is a time seris
                            save_tensor.append(Convert_batch_of_time_series_to_batch_of_img_torch_array(x.cpu(), y, masks=mask.repeat(1, 1, 1)))
                        else:
                            raise NotImplementedError("Unknown mask shape.")

                    # path = os.path.join(self.path, self.umap_placeholder.format(trainer.current_epoch, n))
                    path = os.path.join(self.path, f"ep_{str(trainer.current_epoch).zfill(3)}-batch_{str(n).zfill(5)}.png")
                    if len(x.shape) == 4:
                        save_tensor_cat = torch.cat(save_tensor).float()
                    elif len(x.shape) == 3: # Time series
                        save_tensor_cat = torch.cat(save_tensor).float() / 255.
                    else:
                        raise NotImplementedError("Unknown data shape.")

                    img_list = save_image_with_return_img_list(save_tensor_cat, path)
                    # print(f"len(img_list): {len(img_list)}")
                    wandb.log({"examples": [wandb.Image(img) for img in img_list]}, step=trainer.global_step)


            module.train()

    def on_validation_end(self, trainer: pl.Trainer, module: pl.LightningModule):
        """Tries to generate an up-to-date UMAP visualization of the features
        at the end of each validation epoch.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        """ https://github.com/Lightning-AI/lightning/issues/11054 
        Solves issues that val_dataloader is not shuffled when using 'ddp'
        """

        epoch = trainer.current_epoch  # type: ignore
        if epoch % self.frequency == 0 and not trainer.sanity_checking:
            self.plot(trainer, module)
