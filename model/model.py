from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch_dataset import masked_mse_loss


class LSTMAutoencoder(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 32,
        num_layers: int = 1,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()  # W&B에 자동으로 하이퍼파라미터 기록

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        ...
        # Note by Karyx💫: This code is omitted to protect my intellectual property.

    def _step(self, batch, stage: str):
        padded, lengths, mask, norad_ids = batch
        reconstructed = self(padded, lengths)
        loss = masked_mse_loss(reconstructed, padded, mask)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=padded.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
