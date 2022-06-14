# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch import Tensor

from vam.conformer.encoder import ConformerEncoder
import torch.nn.functional as F
import pytorch_lightning as pl
from vam.models.wav_decoder import Generator, AudioEncoder


class Conformer(pl.LightningModule):
    """
    Conformer: Convolution-augmented Transformer for Speech Recognition
    The paper used a one-lstm Transducer decoder, currently still only implemented
    the conformer encoder shown in the paper.

    Args:
        num_classes (int): Number of classification classes
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of conformer encoder
        decoder_dim (int, optional): Dimension of conformer decoder
        num_encoder_layers (int, optional): Number of conformer blocks
        num_decoder_layers (int, optional): Number of decoder layers
        decoder_rnn_type (str, optional): type of RNN cell
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        decoder_dropout_p (float, optional): Probability of conformer decoder dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by conformer.
        - **output_lengths** (batch): list of sequence output lengths
    """
    def __init__(
            self,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_encoder_layers: int = 17,
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
            decode_wav=False,
            encode_wav=False,
            encoder_ratios="8,4,2,2",
            decoder_ratios="8,4,2,2",
            encoder_residual_layers=3,
            decoder_residual_layers=3,
            ngf=32,
            use_visual_pe=False
    ) -> None:
        super(Conformer, self).__init__()
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            crossmodal_dropout_p=crossmodal_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
            use_crossmodal_layer=use_crossmodal_layer,
            use_visual_pe=use_visual_pe
        )
        if encode_wav:
            self.audio_encoder = AudioEncoder(input_size=1, ngf=ngf, n_residual_layers=encoder_residual_layers,
                                              ratios=encoder_ratios)
        else:
            self.audio_encoder = None
        if decode_wav:
            self.decoder = Generator(input_size=512, ngf=ngf, n_residual_layers=decoder_residual_layers,
                                     ratios=decoder_ratios)
        else:
            self.decoder = nn.Linear(encoder_dim, input_dim)

    def set_encoder(self, encoder):
        """ Setter for encoder """
        self.encoder = encoder

    def set_decoder(self, decoder):
        """ Setter for decoder """
        self.decoder = decoder

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        num_audio_encoder_parameters = self.audio_encoder.count_parameters()
        num_encoder_parameters = self.encoder.count_parameters()
        num_decoder_parameters = self.decoder.count_parameters()
        num_stop_classifier_parameters = self.rt60_predictor.count_parameters()
        return num_audio_encoder_parameters + num_encoder_parameters + num_decoder_parameters + num_stop_classifier_parameters

    def update_dropout(self, dropout_p) -> None:
        """ Update dropout probability of model """
        self.audio_encoder.update_dropout(dropout_p)
        self.encoder.update_dropout(dropout_p)
        self.decoder.update_dropout(dropout_p)
        self.rt60_predictor.update_dropout(dropout_p)

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor = None,
            img_feat = None
    ):
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            target_lengths (torch.LongTensor): The length of target tensor. ``(batch)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        if self.audio_encoder is not None:
            audio_feat = self.audio_encoder(inputs).permute(0, 2, 1)
        encoder_outputs = self.encoder(audio_feat, input_lengths, img_feat=img_feat)
        decoder_outputs = self.decoder(encoder_outputs)
        return decoder_outputs, None

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        tgt_pred, tgt = self.dereverb(batch, batch_idx)
        loss = F.mse_loss(tgt_pred, tgt)

        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        tgt_pred, tgt = self.dereverb(batch, batch_idx)
        loss = F.mse_loss(tgt_pred, tgt)

        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        return loss

    def dereverb(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        for key, value in batch.items():
            batch[key] = value.to(device=self.device, dtype=torch.float)
        receiver_spec = batch['recv_spec']
        source_spec = batch['src_spec']

        receiver_spec[..., 0] = torch.log1p(receiver_spec[..., 0])
        source_spec[..., 0] = torch.log1p(source_spec[..., 0])

        # seq, batch, dim
        cropped_receiver_spec = receiver_spec[:, :, :, 0].permute(0, 2, 1)
        cropped_source_spec = source_spec[:, :, :, 0].permute(0, 2, 1)

        src = cropped_receiver_spec
        # batch, seq, dim
        tgt_pred = self.forward(src, torch.ones((src.size(1)) * 257))

        return tgt_pred, cropped_source_spec

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
