# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from vam.conformer.feed_forward import FeedForwardModule
from vam.conformer.attention import MultiHeadedSelfAttentionModule
from vam.models.vit import MultiHeadedCrossmodalAttentionModule
# from vam.conformer.attention import MultiHeadedCrossmodalAttentionModule
from vam.conformer.convolution import (
    ConformerConvModule,
)
from vam.conformer.modules import (
    ResidualConnectionModule,
    Linear,
)
import pytorch_lightning as pl


class ConformerBlock(pl.LightningModule):
    """
    Conformer block contains two Feed Forward modules sandwiching the Multi-Headed Self-Attention module
    and the Convolution module. This sandwich structure is inspired by Macaron-Net, which proposes replacing
    the original feed-forward layer in the Transformer block into two half-step feed-forward layers,
    one before the attention layer and one after.

    Args:
        encoder_dim (int, optional): Dimension of conformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by conformer block.
    """
    def __init__(
            self,
            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            crossmodal_dropout_p=0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
            use_crossmodal_layer=False,
            use_visual_pe=False
    ):
        super(ConformerBlock, self).__init__()
        self.use_crossmodal_layer = use_crossmodal_layer
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        modules = []
        modules.append(ResidualConnectionModule(
                    module=FeedForwardModule(
                        encoder_dim=encoder_dim,
                        expansion_factor=feed_forward_expansion_factor,
                        dropout_p=feed_forward_dropout_p
                    ),
                    module_factor=self.feed_forward_residual_factor,
                ))
        modules.append(ResidualConnectionModule(
                    module=MultiHeadedSelfAttentionModule(
                        d_model=encoder_dim,
                        num_heads=num_attention_heads,
                        dropout_p=attention_dropout_p
                    ),
                ))
        if use_crossmodal_layer:
            modules.append(ResidualConnectionModule(
                    module=MultiHeadedCrossmodalAttentionModule(
                        d_model=encoder_dim,
                        num_heads=num_attention_heads,
                        dropout_p=crossmodal_dropout_p,
                        use_visual_pe=use_visual_pe
                    ),
                ))
        modules.append(ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p
                ),
            ))
        modules.append(ResidualConnectionModule(
                    module=FeedForwardModule(
                        encoder_dim=encoder_dim,
                        expansion_factor=feed_forward_expansion_factor,
                        dropout_p=feed_forward_dropout_p
                    ),
                    module_factor=self.feed_forward_residual_factor,
                ))
        modules.append(nn.LayerNorm(encoder_dim))
        self.sequential = nn.Sequential(*modules)

    def forward(self, inputs: Tensor, img_feat=None) -> Tensor:
        if self.use_crossmodal_layer:
            x = self.sequential[0](inputs)
            x = self.sequential[1](x)
            x = self.sequential[2](x, img_feat=img_feat)
            for layer in self.sequential[3:]:
                x = layer(x)
            return x
        else:
            return self.sequential(inputs.to(self.device))


class ConformerEncoder(pl.LightningModule):
    """
    Conformer encoder first processes the input with a convolution subsampling layer and then
    with a number of conformer blocks.

    Args:
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of conformer encoder
        num_layers (int, optional): Number of conformer blocks
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by conformer encoder.
        - **output_lengths** (batch): list of sequence output lengths
    """
    def __init__(
            self,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            crossmodal_dropout_p=0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
            use_crossmodal_layer=False,
            use_visual_pe=False
    ):
        super(ConformerEncoder, self).__init__()
        # self.conv_subsample = Conv2dSubampling(in_channels=1, out_channels=encoder_dim)
        self.input_projection = nn.Sequential(
            Linear(input_dim, encoder_dim),
            nn.Dropout(p=input_dropout_p),
        )
        self.layers = nn.ModuleList([ConformerBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            crossmodal_dropout_p=crossmodal_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            use_crossmodal_layer=use_crossmodal_layer,
            use_visual_pe=use_visual_pe
        ) for _ in range(num_layers)])

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor, input_lengths: Tensor = None, img_feat=None) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  encoder training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor)

            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        # outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.input_projection(inputs)

        for layer in self.layers:
            outputs = layer(outputs, img_feat)

        return outputs
