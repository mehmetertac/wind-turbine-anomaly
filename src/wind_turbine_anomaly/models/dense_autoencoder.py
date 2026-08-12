"""Dense (MLP) autoencoder for multivariate SCADA anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from wind_turbine_anomaly.config import (
    AE_BATCH_SIZE,
    AE_BOTTLENECK_DIM,
    AE_EPOCHS,
    AE_HIDDEN_DIMS,
    AE_LR,
    AE_PATIENCE,
    AE_VAL_FRACTION,
)


class _DenseAutoencoder(nn.Module):
    """MLP encoder-decoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        bottleneck_dim: int,
    ) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        prev = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev, dim), nn.ReLU()])
            prev = dim
        encoder_layers.append(nn.Linear(prev, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        prev = bottleneck_dim
        for dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev, dim), nn.ReLU()])
            prev = dim
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@dataclass
class DenseAutoencoderPipeline:
    """Scaler + dense autoencoder fitted on healthy data."""

    scaler: StandardScaler
    model: _DenseAutoencoder
    feature_columns: list[str]
    device: torch.device

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Return per-row reconstruction MSE (higher = more anomalous)."""
        self.model.eval()
        X_arr = self.scaler.transform(X[self.feature_columns].values)
        with torch.no_grad():
            tensor = torch.tensor(X_arr, dtype=torch.float32, device=self.device)
            recon = self.model(tensor)
            mse = ((recon - tensor) ** 2).mean(dim=1).cpu().numpy()
        return pd.Series(mse, index=X.index, name="anomaly_score")


def _train_with_early_stopping(
    model: _DenseAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0

    for _ in range(epochs):
        model.train()
        for batch, in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch, in val_loader:
                batch = batch.to(device)
                val_loss += criterion(model(batch), batch).item()
                n_batches += 1
        val_loss /= max(n_batches, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def fit_dense_autoencoder(
    X_train: pd.DataFrame,
    feature_columns: list[str],
    hidden_dims: list[int] | None = None,
    bottleneck_dim: int = AE_BOTTLENECK_DIM,
    epochs: int = AE_EPOCHS,
    batch_size: int = AE_BATCH_SIZE,
    lr: float = AE_LR,
    patience: int = AE_PATIENCE,
    val_fraction: float = AE_VAL_FRACTION,
    random_state: int = 42,
    device: torch.device | None = None,
) -> DenseAutoencoderPipeline:
    """Fit StandardScaler and dense autoencoder on healthy training rows."""
    hidden_dims = hidden_dims or AE_HIDDEN_DIMS
    device = device or torch.device("cpu")
    torch.manual_seed(random_state)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train[feature_columns].values)

    n_val = max(1, int(len(X_scaled) * val_fraction))
    n_train = len(X_scaled) - n_val
    train_arr = X_scaled[:n_train]
    val_arr = X_scaled[n_train:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_arr, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_arr, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )

    input_dim = len(feature_columns)
    model = _DenseAutoencoder(input_dim, hidden_dims, bottleneck_dim).to(device)
    _train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        patience=patience,
        device=device,
    )

    return DenseAutoencoderPipeline(
        scaler=scaler,
        model=model,
        feature_columns=feature_columns,
        device=device,
    )
