"""LSTM autoencoder for sequence-based SCADA anomaly detection."""

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
    AE_EPOCHS,
    AE_LR,
    AE_PATIENCE,
    AE_VAL_FRACTION,
    LSTM_EPOCHS,
    LSTM_HIDDEN_DIM,
    LSTM_LATENT_DIM,
    LSTM_MAX_TRAIN_WINDOWS,
    LSTM_WINDOW_SIZE,
)
from wind_turbine_anomaly.models.sequences import build_sliding_windows


class _LSTMAutoencoder(nn.Module):
    """Sequence encoder-decoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)
        latent = self.to_latent(h_n[-1])
        hidden = self.from_latent(latent)
        seq_len = x.size(1)
        dec_in = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(dec_in)
        return self.output(dec_out)


@dataclass
class LSTMAutoencoderPipeline:
    """Scaler + LSTM autoencoder fitted on healthy data."""

    scaler: StandardScaler
    model: _LSTMAutoencoder
    feature_columns: list[str]
    window_size: int
    device: torch.device

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Return window reconstruction MSE at each window end timestamp."""
        self.model.eval()
        X_arr = self.scaler.transform(X[self.feature_columns].values)
        windows, end_indices = build_sliding_windows(X_arr, self.window_size)
        if len(windows) == 0:
            return pd.Series(dtype=float, name="anomaly_score")

        with torch.no_grad():
            batch_size = 512
            mse_parts: list[np.ndarray] = []
            for start in range(0, len(windows), batch_size):
                chunk = windows[start : start + batch_size]
                tensor = torch.tensor(chunk, dtype=torch.float32, device=self.device)
                recon = self.model(tensor)
                mse_parts.append(
                    ((recon - tensor) ** 2).mean(dim=(1, 2)).cpu().numpy()
                )
            mse = np.concatenate(mse_parts)

        end_times = X.index[end_indices]
        return pd.Series(mse, index=end_times, name="anomaly_score")


def _train_with_early_stopping(
    model: _LSTMAutoencoder,
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


def fit_lstm_autoencoder(
    X_train: pd.DataFrame,
    feature_columns: list[str],
    window_size: int = LSTM_WINDOW_SIZE,
    hidden_dim: int = LSTM_HIDDEN_DIM,
    latent_dim: int = LSTM_LATENT_DIM,
    epochs: int = LSTM_EPOCHS,
    batch_size: int = AE_BATCH_SIZE,
    lr: float = AE_LR,
    patience: int = AE_PATIENCE,
    val_fraction: float = AE_VAL_FRACTION,
    max_train_windows: int | None = LSTM_MAX_TRAIN_WINDOWS,
    random_state: int = 42,
    device: torch.device | None = None,
) -> LSTMAutoencoderPipeline:
    """Fit StandardScaler and LSTM autoencoder on healthy training rows."""
    device = device or torch.device("cpu")
    torch.manual_seed(random_state)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train[feature_columns].values)
    windows, _ = build_sliding_windows(X_scaled, window_size)
    if len(windows) < 10:
        raise ValueError(
            f"Insufficient training windows ({len(windows)}); need at least 10 "
            f"for window_size={window_size}"
        )

    if max_train_windows is not None and len(windows) > max_train_windows:
        rng = np.random.default_rng(random_state)
        keep = np.sort(rng.choice(len(windows), max_train_windows, replace=False))
        windows = windows[keep]

    n_val = max(1, int(len(windows) * val_fraction))
    n_train = len(windows) - n_val
    train_windows = windows[:n_train]
    val_windows = windows[n_train:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_windows, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_windows, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )

    input_dim = len(feature_columns)
    model = _LSTMAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
    _train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        patience=patience,
        device=device,
    )

    return LSTMAutoencoderPipeline(
        scaler=scaler,
        model=model,
        feature_columns=feature_columns,
        window_size=window_size,
        device=device,
    )
