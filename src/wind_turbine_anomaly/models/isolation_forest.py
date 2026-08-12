"""Isolation Forest baseline for multivariate SCADA anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from wind_turbine_anomaly.models.scoring import save_scores, threshold_from_training


@dataclass
class IsolationForestPipeline:
    """Scaler + IsolationForest fitted on healthy data."""

    scaler: StandardScaler
    model: IsolationForest
    feature_columns: list[str]

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Return anomaly scores (higher = more anomalous)."""
        X_arr = self.scaler.transform(X[self.feature_columns].values)
        # Negate decision_function so higher values indicate stronger anomalies.
        raw = -self.model.decision_function(X_arr)
        return pd.Series(raw, index=X.index, name="anomaly_score")


def fit_isolation_forest(
    X_train: pd.DataFrame,
    feature_columns: list[str],
    contamination: float = 0.01,
    random_state: int = 42,
    n_estimators: int = 200,
) -> IsolationForestPipeline:
    """Fit StandardScaler and IsolationForest on healthy training rows."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train[feature_columns].values)
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
    )
    model.fit(X_scaled)
    return IsolationForestPipeline(
        scaler=scaler,
        model=model,
        feature_columns=feature_columns,
    )


__all__ = [
    "IsolationForestPipeline",
    "fit_isolation_forest",
    "save_scores",
    "threshold_from_training",
]
