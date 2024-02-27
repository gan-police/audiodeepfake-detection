"""Models for classification of audio deepfakes."""

import ast
import sys
from copy import copy
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchsummary import summary

import timm
from timm.models.layers import to_2tuple, trunc_normal_

from .utils import DotDict


def compute_parameter_total(net: torch.nn.Module) -> int:
    """Compute the parameter total of the input net.

    Args:
        net (torch.nn.Module): The model containing the
            parameters to count.

    Returns:
        int: The parameter total.
    """
    total = 0
    for p_name, p in net.named_parameters():
        if p.requires_grad:
            print(p_name)
            print(p.shape)
            total += int(np.prod(p.shape))  # type: ignore
    return total


class GridModelWrapper(nn.Module):
    """Deep CNN."""

    def __init__(
        self,
        sequentials: list,
        transforms: list,
    ) -> None:
        """Define network sturcture."""
        super(GridModelWrapper, self).__init__()

        self.sequentials = nn.ParameterList(sequentials)
        self.transforms = transforms
        self.len = len(self.sequentials)

        if len(self.transforms) != self.len:
            print("Warning: length of transforms and sequentials are not the same.")

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        for i in range(self.len):
            x = self.sequentials[i](x)
            if len(self.transforms) > i:
                for j in range(len(self.transforms[i])):
                    x = self.transforms[i][j](x)

        return x


class LCNN(nn.Module):
    """Deep CNN with 2D convolutions for detecting audio deepfakes.

    Fork of ASVSpoof Challenge 2021 LA Baseline.
    """

    def __init__(
        self,
        classes: int = 2,
        in_channels: int = 1,
        lstm_channels: int = 256,
    ) -> None:
        """Define network sturcture."""
        super(LCNN, self).__init__()

        # LCNN from AVSpoofChallenge 2021
        self.lcnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, 1, padding=2),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 1, 1, padding=0),
            MaxFeatureMap2D(),
            nn.SyncBatchNorm(32, affine=False),
            nn.Conv2d(32, 96, 3, 1, padding=1),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2, 2),
            nn.SyncBatchNorm(48, affine=False),
            nn.Conv2d(48, 96, 1, 1, padding=0),
            MaxFeatureMap2D(),
            nn.SyncBatchNorm(48, affine=False),
            nn.Conv2d(48, 128, 3, 1, padding=1),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 1, 1, padding=0),
            MaxFeatureMap2D(),
            nn.SyncBatchNorm(64, affine=False),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            MaxFeatureMap2D(),
            nn.SyncBatchNorm(32, affine=False),
            nn.Conv2d(32, 64, 1, 1, padding=0),
            MaxFeatureMap2D(),
            nn.SyncBatchNorm(32, affine=False),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            MaxFeatureMap2D(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.7),
        )

        self.lstm = nn.Sequential(
            BLSTMLayer((lstm_channels // 16) * 32, (lstm_channels // 16) * 32),
            BLSTMLayer((lstm_channels // 16) * 32, (lstm_channels // 16) * 32),
        )

        self.fc = nn.Linear((lstm_channels // 16) * 32, classes)

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        x = self.lcnn(x.permute(0, 1, 3, 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        shape = x.shape
        x = self.lstm(x.view(shape[0], shape[1], -1))
        x = self.fc(x).mean(1)

        return x


class Regression(torch.nn.Module):
    """A shallow linear-regression model."""

    def __init__(self, args: DotDict) -> None:
        """Create the regression model.

        Args:
            args (DotDict): The configuration dictionary.
        """
        super().__init__()
        self.linear = torch.nn.Linear(args.num_of_scales * 101, 2)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the regression forward pass.

        Args:
            x (torch.Tensor): An input tensor of shape [batch_size, ...].

        Returns:
            torch.Tensor: A logsoftmax scaled output of shape
                [batch_size, classes].
        """
        x_flat = torch.reshape(x, [x.shape[0], -1])
        return self.logsoftmax(self.linear(x_flat))


class MaxFeatureMap2D(nn.Module):
    """Max feature map (along 2D).

    MaxFeatureMap2D(max_dim=1) from AVSPoofChallenge 2021 LA Baseline.

    l_conv2d = MaxFeatureMap2D(1)
    data_in = torch.rand([1, 4, 5, 5])
    data_out = l_conv2d(data_in)

    Input:
    ------
    data_in: tensor of shape (batch, channel, ...)

    Output:
    -------
    data_out: tensor of shape (batch, channel//2, ...)

    Note
    ----
    By default, Max-feature-map is on channel dimension,
    and maxout is used on (channel ...)
    """

    def __init__(self, max_dim: int = 1) -> None:
        """Initialize with max feature dimension."""
        super().__init__()
        self.max_dim = max_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward max feature map."""
        # suppose inputs (batchsize, channel, length, dim)

        shape = list(inputs.size())

        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)

        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, _ = inputs.reshape(*shape).max(self.max_dim)
        return m


class BLSTMLayer(nn.Module):
    """Wrapper over dilated conv1D.

    From AVSPoofChallenge 2021 LA Baseline.

    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)
    We want to keep the length the same.
    """

    def __init__(self, input_dim, output_dim) -> None:
        """Initialize class."""
        super(BLSTMLayer, self).__init__()
        if output_dim % 2 != 0:
            print("Output_dim of BLSTMLayer is {:d}".format(output_dim))
            print("BLSTMLayer expects a layer size of even number")
            sys.exit(1)
        # bi-directional LSTM
        self.l_blstm = nn.LSTM(input_dim, output_dim // 2, bidirectional=True)

    def forward(self, x):
        """Forward lstm input."""
        # permute to (length, batchsize=1, dim)
        blstm_data, _ = self.l_blstm(x.permute(1, 0, 2))
        # permute it backt to (batchsize=1, length, dim)
        return blstm_data.permute(1, 0, 2)


class DCNN(torch.nn.Module):
    """Deep CNN with dilated convolutions."""

    def __init__(
        self,
        args: DotDict,
    ) -> None:
        """Define network sturcture.

        Args:
            args (DotDict): The configuration dictionary.
        """
        super(DCNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                args.input_dim[1], args.ochannels1, args.kernel1, stride=1, padding=2
            ),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.SyncBatchNorm(args.ochannels1, affine=False),
            nn.Conv2d(args.ochannels1, args.ochannels2, 1, 1, padding=0),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels2, affine=False),
            nn.Conv2d(args.ochannels2, args.ochannels3, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.SyncBatchNorm(args.ochannels3, affine=False),
            nn.Conv2d(args.ochannels3, args.ochannels4, 3, 1, padding=1),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels4, affine=False),
            nn.Conv2d(args.ochannels4, args.ochannels5, 3, 1, padding=1),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels5, affine=False),
            nn.Conv2d(args.ochannels5, 64, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(args.dropout_cnn),
        )

        time_dim = ((args.input_dim[-1])) // 8 + args.time_dim_add

        self.dil_conv = nn.Sequential(
            nn.SyncBatchNorm(time_dim, affine=True),
            nn.Conv2d(time_dim, time_dim, 3, 1, padding=1, dilation=1),
            nn.PReLU(),
            nn.SyncBatchNorm(time_dim, affine=True),
            nn.Conv2d(time_dim, time_dim, 5, 1, padding=2, dilation=2),
            nn.PReLU(),
            nn.SyncBatchNorm(time_dim, affine=True),
            nn.Conv2d(time_dim, time_dim, 7, 1, padding=2, dilation=4),
            nn.PReLU(),
            nn.Dropout(args.dropout_lstm),
        )

        self.fc = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(args.flattend_size, 2),
        )
        self.single_gpu = not args.ddp

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        # [batch, channels, packets, time]
        x = self.cnn(x.permute(0, 1, 3, 2))

        # [batch, channels, time, packets]
        x = x.permute(0, 2, 1, 3).contiguous()

        # "[batch, time, channels, packets]"
        x = self.dil_conv(x)
        x = self.fc(x).mean(1)

        return x

    def get_name(self) -> str:
        """Get name of model."""
        return "DCNN"


class DCNNxDropout(torch.nn.Module):
    """Deep CNN with dilated convolutions."""

    def __init__(
        self,
        args: DotDict,
    ) -> None:
        """Define network sturcture.

        Args:
            args (DotDict): The configuration dictionary.
        """
        super(DCNNxDropout, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                args.input_dim[1], args.ochannels1, args.kernel1, stride=1, padding=2
            ),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.SyncBatchNorm(args.ochannels1, affine=False),
            nn.Conv2d(args.ochannels1, args.ochannels2, 1, 1, padding=0),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels2, affine=False),
            nn.Conv2d(args.ochannels2, args.ochannels3, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.SyncBatchNorm(args.ochannels3, affine=False),
            nn.Conv2d(args.ochannels3, args.ochannels4, 3, 1, padding=1),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels4, affine=False),
            nn.Conv2d(args.ochannels4, args.ochannels5, 3, 1, padding=1),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels5, affine=False),
            nn.Conv2d(args.ochannels5, 64, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
        )

        time_dim = ((args.input_dim[-1])) // 8 + args.time_dim_add

        self.dil_conv = nn.Sequential(
            nn.SyncBatchNorm(time_dim, affine=True),
            nn.Conv2d(time_dim, time_dim, 3, 1, padding=1, dilation=1),
            nn.PReLU(),
            nn.SyncBatchNorm(time_dim, affine=True),
            nn.Conv2d(time_dim, time_dim, 5, 1, padding=2, dilation=2),
            nn.PReLU(),
            nn.SyncBatchNorm(time_dim, affine=True),
            nn.Conv2d(time_dim, time_dim, 7, 1, padding=2, dilation=4),
            nn.PReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(args.flattend_size, 2),
        )
        self.single_gpu = not args.ddp

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        # [batch, channels, packets, time]
        x = self.cnn(x.permute(0, 1, 3, 2))

        # [batch, channels, time, packets]
        x = x.permute(0, 2, 1, 3).contiguous()

        # "[batch, time, channels, packets]"
        x = self.dil_conv(x)
        x = self.fc(x).mean(1)

        return x

    def get_name(self) -> str:
        """Get name of model."""
        return "DCNNxDropout"


class DCNNxDilation(torch.nn.Module):
    """Deep CNN without dilated convolutions."""

    def __init__(
        self,
        args: DotDict,
    ) -> None:
        """Define network sturcture.

        Args:
            args (DotDict): The configuration dictionary.
        """
        super(DCNNxDilation, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                args.input_dim[1], args.ochannels1, args.kernel1, stride=1, padding=2
            ),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.SyncBatchNorm(args.ochannels1, affine=False),
            nn.Conv2d(args.ochannels1, args.ochannels2, 1, 1, padding=0),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels2, affine=False),
            nn.Conv2d(args.ochannels2, args.ochannels3, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.SyncBatchNorm(args.ochannels3, affine=False),
            nn.Conv2d(args.ochannels3, args.ochannels4, 3, 1, padding=1),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels4, affine=False),
            nn.Conv2d(args.ochannels4, args.ochannels5, 3, 1, padding=1),
            nn.PReLU(),
            nn.SyncBatchNorm(args.ochannels5, affine=False),
            nn.Conv2d(args.ochannels5, 64, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(args.dropout_cnn),
        )

        self.fc = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(args.flattend_size, 2),
        )
        self.single_gpu = not args.ddp

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        # [batch, channels, packets, time]
        x = self.cnn(x.permute(0, 1, 3, 2))

        # [batch, channels, time, packets]
        x = x.permute(0, 2, 1, 3).contiguous()

        # "[batch, time, channels, packets]"
        x = self.fc(x).mean(1)

        return x

    def get_name(self) -> str:
        """Get name of model."""
        return "DCNNxDilation"


class PatchEmbed(nn.Module):
    """Patch embedding to be used in ASTModel.

    Fork from official repo for Audio Spectrogram Transformer (AST) at
    https://github.com/YuanGongND/ast/blob/master/src/models/ast_models.py.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """Init patch embedding."""
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """Forward patch."""
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(nn.Module):
    """AST model.

    Fork from official repo for Audio Spectrogram Transformer (AST) at
    https://github.com/YuanGongND/ast/blob/master/src/models/ast_models.py.
    """

    def __init__(
        self,
        args: DotDict,
        label_dim: int = 2,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 256,
        input_tdim: int = 101,
        imagenet_pretrain: bool = True,
        model_size: str = "base384",
        verbose: bool = True,
    ):
        """Initialize AST model.

        Args:
            args (DotDict): Experiment config.
            label_dim (int): The label dimension, i.e., the number of total classes. Defaults to 2.
            fstride (int): The stride of patch spliting on the frequency dimension,
                           for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap
                           of 6. Defaults to 10.
            tstride (int): The stride of patch spliting on the time dimension, for 16*16 patchs,
                           tstride=16 means no overlap, tstride=10 means overlap of 6. Defaults to 10.
            input_fdim (int): The number of frequency bins of the input spectrogram. Defaults to 256.
            input_tdim (int): The number of time frames of the input spectrogram. Defaults to 101.
            imagenet_pretrain (bool): If use ImageNet pretrained model. Defaults to True.
            model_size (str): The model size of AST, should be in [tiny224, small224, base224, base384],
                              base224 and base 384 are same model, but are trained differently during
                              ImageNet pretraining. Defaults to 'base384'.
            verbose (bool): If verbose output should be shown in detail. Defaults to True.

        Raises:
            Exception: If timm version is not 0.4.5.
        """
        super(ASTModel, self).__init__()
        assert (
            timm.__version__ == "0.4.5"
        ), "Please use timm == 0.4.5, the code might not be compatible with newer versions."

        input_tdim = args.flattend_size

        if verbose:
            print("---------------AST Model Summary---------------")
            print("ImageNet pretraining: {:s}".format(str(imagenet_pretrain)))

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if imagenet_pretrain:
            if model_size == "tiny224":
                self.v = timm.create_model(
                    "vit_deit_tiny_distilled_patch16_224", pretrained=imagenet_pretrain
                )
            elif model_size == "small224":
                self.v = timm.create_model(
                    "vit_deit_small_distilled_patch16_224", pretrained=imagenet_pretrain
                )
            elif model_size == "base224":
                self.v = timm.create_model(
                    "vit_deit_base_distilled_patch16_224", pretrained=imagenet_pretrain
                )
            elif model_size == "base384":
                self.v = timm.create_model(
                    "vit_deit_base_distilled_patch16_384", pretrained=imagenet_pretrain
                )
            else:
                raise Exception(
                    "Model size must be one of tiny224, small224, base224, base384."
                )
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches**0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim),
            )

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print(
                    "frequncey stride={:d}, time stride={:d}".format(fstride, tstride)
                )
                print("number of patches={:d}".format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(
                1,
                self.original_embedding_dim,
                kernel_size=(16, 16),
                stride=(fstride, tstride),
            )
            if imagenet_pretrain:
                new_proj.weight = torch.nn.Parameter(
                    torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1)
                )
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain:
                # get the positional embedding from deit model, skip the first two tokens
                # (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = (
                    self.v.pos_embed[:, 2:, :]
                    .detach()
                    .reshape(1, self.original_num_patches, self.original_embedding_dim)
                    .transpose(1, 2)
                    .reshape(
                        1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw
                    )
                )
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :,
                        :,
                        :,
                        int(self.oringal_hw / 2)
                        - int(t_dim / 2) : int(self.oringal_hw / 2)
                        - int(t_dim / 2)
                        + t_dim,
                    ]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(
                        new_pos_embed, size=(self.oringal_hw, t_dim), mode="bilinear"
                    )
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :,
                        :,
                        int(self.oringal_hw / 2)
                        - int(f_dim / 2) : int(self.oringal_hw / 2)
                        - int(f_dim / 2)
                        + f_dim,
                        :,
                    ]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(
                        new_pos_embed, size=(f_dim, t_dim), mode="bilinear"
                    )
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(
                    1, self.original_embedding_dim, num_patches
                ).transpose(1, 2)
                # concatenate the above positional embedding with the cls token and distillation token
                # of the deit model.
                self.v.pos_embed = nn.Parameter(
                    torch.cat(
                        [self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1
                    )
                )
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable
                # positional embedding
                new_pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        self.v.patch_embed.num_patches + 2,
                        self.original_embedding_dim,
                    )
                )
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=0.02)

    def get_shape(self, fstride, tstride, input_fdim=256, input_tdim=101):
        """Get shape of f_dim and t_dim."""
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(
            1,
            self.original_embedding_dim,
            kernel_size=(16, 16),
            stride=(fstride, tstride),
        )
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x):
        """Forward model input.

        Args:
            x (torch.Tensor): The input spectrogram, expected shape(batch_size, channels,
                              frequency_bins, time_frame_num), e.g. (128, 1, 256, 101).

        Returns:
            torch.Tensor: Prediction.
        """
        b = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(b, -1, -1)
        dist_token = self.v.dist_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        return x

    def get_name(self) -> str:
        """Get name of model."""
        return "AST"


def get_model(
    args: DotDict,
    model_name: str,
    nclasses: int = 2,
    in_channels: int = 1,
    lead: bool = False,
) -> Union[LCNN, GridModelWrapper, Any]:
    """Get torch module model with given parameters.

    Args:
        args (DotDict): Configuration dictionary.
        model_name (str): Model name as str, one of: "lcnn", "gridmodel", "modules".
        nclasses (int): Number of output classes of the network. Defaults to 2.
        in_channels (int): Number of input channels. Defaults to 1.
        lead (bool): True if gpu rank is 0. Defaults to False.

    Raises:
        RuntimeError: If parsed model from model string has faults.
        RuntimeError: If modular model fails in executing torchsummary, e.g. if the input
                      dimensions do not fit into the model layers.
        RuntimeError: If model name is not valid.
        RuntimeError: If model name is gridmodel but model_data is not inside the config.

    Returns:
        Union[LCNN, GridModelWrapper, Any]: The model or model wrapper.
    """
    model: LCNN | GridModelWrapper | Any = None
    if model_name == "lcnn":
        if "doubledelta" in args.features:
            lstm_channels = 60
        elif "delta" in args.features:
            lstm_channels = 40
        elif "lfcc" in args.features:
            lstm_channels = 20
        else:
            lstm_channels = int(args.num_of_scales)
        model = LCNN(
            classes=nclasses,
            in_channels=in_channels,
            lstm_channels=lstm_channels,
        )
    elif model_name == "gridmodel":
        if not hasattr(args, "model_data"):
            raise RuntimeError(
                "Config dict does not contain the key model_data,"
                "which should hold the list like model structure."
            )
        model = get_gridsearch_model(args.model_data)
    elif model_name == "modules":
        if check_dimensions(args.module(args), args.input_dim[1:], verbose=lead):
            model = args.module(args)
        else:
            raise RuntimeError("Model not valid.")
    else:
        raise RuntimeError(f"Model with model string '{model_name}' does not exist.")
    return model


def get_gridsearch_model(model_data: list) -> GridModelWrapper:
    """Generate sequential torch model from list like structure.

    Args:
        model_data (list): A model given in a list like structure.

    Raises:
        RuntimeError: If model_data has incorrect format.

    Returns:
        GridModelWrapper: The wrapper class for the sequential grid model.
    """
    # parse model data if exists
    model_data = parse_model(model_data)

    model_seq = []
    transforms = []
    for model_layer in model_data:
        if "input_shape" in model_layer.keys():
            input_shape = model_layer["input_shape"]
        else:
            input_shape = None

        if "transforms" in model_layer.keys():
            transform = model_layer["transforms"]
        else:
            transform = []

        model_seq.append(
            parse_sequential(model_list=model_layer["layers"], input_shape=input_shape)
        )
        transforms.append(transform)

    if False not in model_seq:
        return GridModelWrapper(
            sequentials=model_seq,
            transforms=transforms,
        )
    else:
        raise RuntimeError("Model not valid.")


def parse_model(model_data: list) -> list:
    """Parse the model data list into a structured list.

    Examples:
        Example model string structure:

        >>> model_data = [[
        ...                 {
        ...                     "layers": [
        ...                         [torchvision.ops, "Permute 0,1,3,2"],
        ...                         "Conv2d 1 [64,32,128] 2 1 2",
        ...                         "MaxFeatureMap2D",
        ...                         "MaxPool2d 2 2",
        ...                         ...
        ...                         "Dropout 0.7",
        ...                     ],
        ...                     "input_shape": (1, 256, 101),
        ...                     "transforms": [partial(transf)],
        ...                 },
        ...                 {
        ...                     "layers": [
        ...                         "BLSTMLayer 512 512",
        ...                         "BLSTMLayer 512 512",
        ...                         "Dropout 0.1",
        ...                         "Linear 512 2",
        ...                     ],
        ...                     "input_shape": (1, 512),
        ...                     "transforms": [partial(torch.Tensor.mean, dim=1)],
        ...                 },
        ...             ]]

    Args:
        model_data (list): The whole model data.

    Raises:
        RuntimeError: If a parsing error occurs.

    Returns:
        list: The parsed list.
    """
    for i in range(len(model_data)):
        new_els: list[Any] = []
        for j in range(len(model_data[i])):
            trials = parse_model_str(model_data[i][j]["layers"])
            model_data[i][j]["layers"] = trials[0]
            if len(trials) > 1:
                for k in range(1, len(trials)):
                    if len(new_els) < len(trials) - 1:
                        config_copy = [
                            copy(config_part) for config_part in model_data[i]
                        ]
                        config_copy[j]["layers"] = trials[k]
                        new_els.append(config_copy)
                    elif len(new_els) == len(trials) - 1:
                        new_els[k - 1][j]["layers"] = trials[k]
                    else:
                        raise RuntimeError("Parsing error")
            elif len(new_els) > 0:
                for k in range(0, len(new_els)):
                    new_els[k][j]["layers"] = trials[0]
        model_data.extend(new_els)

    return model_data


def parse_model_str(model_str: list) -> list:
    """Parse a given model string into a separated list for each layer.

    Examples:
        Example model string structure:

        >>> model_str = [
        ...                 [torchvision.ops, "Permute 0,1,3,2"],
        ...                 "Conv2d 1 [64,32,128] 2 1 2",
        ...                 "MaxPool2d 2 2",
        ...                 "Conv2d [32,16,64] 64 1 1 0",
        ...                 ...
        ...                 "MaxPool2d 2 2",
        ...                 "Dropout 0.7",
        ...             ]

    Args:
        model_str (list): A list matching the above example structure.

    Raises:
        RuntimeError: If an element is not of type string or list.
        RuntimeError: If the model layers do not contain the same amount of elements
                      when using nested lists.

    Returns:
        list: The parsed result.
    """
    # TODO: write a test for this
    parsed_output: list = []
    postfix: Any = None
    for element in model_str:
        new_elements = []
        output_els = 1
        postfix = None
        if isinstance(element, list):
            postfix = element[0]
            element = element[-1]  # bcause element = [module, layer]
        if isinstance(element, str):
            split = element.split()
            element_parts = [
                ast.literal_eval(part) for part in split[1:]
            ]  # assuming pattern: "Class args"
            element_parts.insert(0, split[0])
        else:
            raise RuntimeError(f"Model string invalid at {element}.")

        # get number of combinations. Must be the same for each layer element.
        for part in element_parts:
            if isinstance(part, list):
                if output_els == 1:
                    output_els = len(part)
                    break

        for i in range(output_els):
            output_list: list[Any] = []

            for part in element_parts:
                if isinstance(part, list):
                    if output_els != len(part):
                        raise RuntimeError(
                            "Model layers must contain the same amount of elements."
                            + f"Expected {output_els}, but got {len(part)}."
                        )
                    part = part[i]
                output_list.append(str(part).replace(" ", ""))
            if postfix is not None:
                output_list = [postfix, output_list]
            new_elements.append(output_list)

        if len(parsed_output) > 0:
            last_layer = copy(parsed_output[-1])
        else:
            last_layer = None

        for i in range(len(new_elements)):
            if len(parsed_output) == 0:
                parsed_output = [[new_elements[i]]]
            elif len(parsed_output) < i + 1:
                if last_layer is not None:
                    layer = copy(last_layer)
                    layer.append(new_elements[i])
                else:
                    layer = [new_elements[i]]
                parsed_output.append(layer)
            else:
                if len(new_elements) == 1:
                    for part in parsed_output:
                        part.append(new_elements[i])
                else:
                    parsed_output[i].append(new_elements[i])

    return parsed_output


def parse_sequential(model_list, input_shape=None) -> nn.Sequential | bool:
    """Parse given model into torch.nn.Module."""
    layers = []

    for layer in model_list:
        if not isinstance(layer[0], str):
            module = layer[0]
            layer_parts = layer[1]
        else:
            module = nn  # default
            layer_parts = layer
        try:
            layer_type = getattr(module, layer_parts[0])
        except AttributeError:
            if layer_parts[0] == "MaxFeatureMap2D":
                layer_type = MaxFeatureMap2D
            elif layer_parts[0] == "BLSTMLayer":
                layer_type = BLSTMLayer
            else:
                print(f"Warning: given layer type {layer_parts[0]} not found.")
                return False

        layer_args = [ast.literal_eval(part) for part in layer_parts[1:]]
        layer = layer_type(*layer_args)
        layers.append(layer)

    model = nn.Sequential(*layers)

    # only perform dim check if input_shape is given
    if input_shape is not None:
        dim_check = check_dimensions(model, input_shape)
        if not dim_check:
            return False

    return model


def check_dimensions(
    model,
    input_shape,
    verbose: bool = True,
) -> bool:
    """Check if model is valid for given dimensions."""
    try:
        summary(model, input_shape, verbose=1 if verbose else 0)
    except RuntimeError as e:
        if verbose:
            print(f"Error: {e}")
        return False
    return True


if __name__ == "__main__":
    input_shape = (1, 256, 101)

    model_data = [
        "Conv2d 1 32 3 2",
        "Conv2d 32 64 3 1",
        "Flatten",
        "Linear 96768 2",
        "ReLU",
        "Softmax 1",
    ]

    parse_sequential(model_data, input_shape)
