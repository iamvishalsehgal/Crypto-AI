"""
LSTM / BiLSTM / GRU time-series forecasting model for crypto trading signals.

Provides:
    - ``LSTMPredictor`` -- a PyTorch ``nn.Module`` that wraps LSTM, BiLSTM, or
      GRU cells followed by a fully-connected classification head
      (BUY / HOLD / SELL).
    - ``LSTMTrainer`` -- high-level training loop with early stopping,
      learning-rate scheduling, model persistence, and sequence preparation.

Usage::

    from crypto_bot.models.lstm_model import LSTMTrainer

    trainer = LSTMTrainer(settings, model_type='bilstm')
    X_train, y_train, X_val, y_val = trainer.prepare_sequences(features_df)
    history = trainer.train(X_train, y_train, X_val, y_val, epochs=80)
    preds = trainer.predict(X_test)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 3  # BUY, HOLD, SELL
ACTION_LABELS = {0: "BUY", 1: "HOLD", 2: "SELL"}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LSTMPredictor(nn.Module):
    """Recurrent classifier for crypto trading signals.

    Supports LSTM, Bidirectional LSTM, and GRU cells selected via
    *cell_type* and *bidirectional* flags.

    Args:
        input_size: Number of features per time-step.
        hidden_size: Hidden units in each recurrent layer.
        num_layers: Number of stacked recurrent layers.
        dropout: Dropout probability applied between recurrent layers and
            before the output projection.
        bidirectional: If ``True``, use bidirectional processing.
        cell_type: One of ``'lstm'`` or ``'gru'``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        cell_type: str = "lstm",
    ) -> None:
        super().__init__()

        cell_type = cell_type.lower()
        if cell_type not in ("lstm", "gru"):
            raise ValueError(f"Unsupported cell_type '{cell_type}'. Choose 'lstm' or 'gru'.")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_type = cell_type

        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        fc_input_size = hidden_size * direction_factor

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, fc_input_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_input_size // 2, NUM_CLASSES),
        )

        self._init_weights()

    # ---- helpers -----------------------------------------------------------
    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for fully-connected layers."""
        for module in self.fc:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            Logits of shape ``(batch, NUM_CLASSES)``.
        """
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden * dir)
        # Take the output of the last time-step.
        last_hidden = rnn_out[:, -1, :]  # (batch, hidden * dir)
        out = self.dropout(last_hidden)
        logits = self.fc(out)  # (batch, NUM_CLASSES)
        return logits


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class LSTMTrainer:
    """High-level wrapper for training and predicting with :class:`LSTMPredictor`.

    Args:
        settings: Global project settings.
        model_type: One of ``'lstm'``, ``'bilstm'``, ``'gru'``.
    """

    VALID_MODEL_TYPES = ("lstm", "bilstm", "gru")

    def __init__(self, settings: Settings, model_type: str = "lstm") -> None:
        model_type = model_type.lower()
        if model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {self.VALID_MODEL_TYPES}, got '{model_type}'."
            )

        self.settings = settings
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[LSTMPredictor] = None

        # Hyper-parameters sourced from settings with sensible fallbacks.
        self.hidden_size: int = getattr(settings.model, "lstm_units", 128)
        self.num_layers: int = getattr(settings.model, "lstm_layers", 2)
        self.learning_rate: float = getattr(settings.model, "ddqn_learning_rate", 1e-3)
        self.batch_size: int = getattr(settings.model, "batch_size", 64)

        logger.info(
            "LSTMTrainer initialised (type=%s, device=%s, hidden=%d, layers=%d)",
            model_type,
            self.device,
            self.hidden_size,
            self.num_layers,
        )

    # ---- data preparation --------------------------------------------------
    def prepare_sequences(
        self,
        data: pd.DataFrame,
        sequence_length: int = 60,
        target_col: str = "signal",
        val_ratio: float = 0.2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a feature DataFrame into overlapping sequences for training.

        The method scales the feature columns, constructs sliding windows of
        length *sequence_length*, and performs a chronological train /
        validation split (no shuffling).

        Args:
            data: DataFrame with feature columns and a *target_col* column
                containing integer class labels (0=BUY, 1=HOLD, 2=SELL).
            sequence_length: Number of past time-steps per sample.
            target_col: Name of the target column.
            val_ratio: Fraction of samples reserved for validation.

        Returns:
            ``(X_train, y_train, X_val, y_val)`` as PyTorch tensors.
        """
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        feature_cols = [c for c in data.columns if c != target_col]
        features = data[feature_cols].values.astype(np.float32)
        targets = data[target_col].values.astype(np.int64)

        # Fit scaler on the entire dataset (before windowing) -- the caller is
        # responsible for ensuring this is the *training* partition only when
        # used inside a walk-forward loop.
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(features)

        # Build sliding windows.
        X_seqs: List[np.ndarray] = []
        y_seqs: List[int] = []
        for i in range(sequence_length, len(features)):
            X_seqs.append(features[i - sequence_length : i])
            y_seqs.append(targets[i])

        X = np.array(X_seqs, dtype=np.float32)
        y = np.array(y_seqs, dtype=np.int64)

        # Time-series split (chronological -- no shuffle).
        split_idx = int(len(X) * (1.0 - val_ratio))
        X_train = torch.tensor(X[:split_idx], dtype=torch.float32)
        y_train = torch.tensor(y[:split_idx], dtype=torch.long)
        X_val = torch.tensor(X[split_idx:], dtype=torch.float32)
        y_val = torch.tensor(y[split_idx:], dtype=torch.long)

        logger.info(
            "Sequences prepared: train=%d, val=%d, seq_len=%d, features=%d",
            len(X_train),
            len(X_val),
            sequence_length,
            X_train.shape[-1],
        )
        return X_train, y_train, X_val, y_val

    # ---- training ----------------------------------------------------------
    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int = 100,
        patience: int = 10,
    ) -> Dict[str, List[float]]:
        """Train the model with early stopping and LR scheduling.

        Args:
            X_train: Training features ``(N, seq_len, n_features)``.
            y_train: Training labels ``(N,)``.
            X_val: Validation features.
            y_val: Validation labels.
            epochs: Maximum training epochs.
            patience: Early-stopping patience (epochs without val-loss
                improvement).

        Returns:
            Dictionary with keys ``'train_loss'``, ``'val_loss'``,
            ``'train_acc'``, ``'val_acc'`` -- one value per completed epoch.
        """
        input_size = X_train.shape[-1]
        bidirectional = self.model_type == "bilstm"
        cell_type = "gru" if self.model_type == "gru" else "lstm"

        self.model = LSTMPredictor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.2,
            bidirectional=bidirectional,
            cell_type=cell_type,
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(1, patience // 3), verbose=False
        )

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
        best_val_loss = float("inf")
        best_model_state: Optional[Dict[str, Any]] = None
        epochs_no_improve = 0

        logger.info("Starting training for up to %d epochs (patience=%d).", epochs, patience)

        for epoch in range(1, epochs + 1):
            # -- train -------------------------------------------------------
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item() * X_batch.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            # -- validate ----------------------------------------------------
            self.model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    logits = self.model(X_batch)
                    loss = criterion(logits, y_batch)
                    val_running_loss += loss.item() * X_batch.size(0)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == y_batch).sum().item()
                    val_total += y_batch.size(0)

            val_loss = val_running_loss / val_total
            val_acc = val_correct / val_total

            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if epoch % max(1, epochs // 10) == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  "
                    "train_acc=%.3f  val_acc=%.3f  lr=%.2e",
                    epoch,
                    epochs,
                    train_loss,
                    val_loss,
                    train_acc,
                    val_acc,
                    optimizer.param_groups[0]["lr"],
                )

            # -- early stopping ----------------------------------------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping at epoch %d (best val_loss=%.4f).", epoch, best_val_loss
                    )
                    break

        # Restore best weights.
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Restored best model weights (val_loss=%.4f).", best_val_loss)

        return history

    # ---- prediction --------------------------------------------------------
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Return class probabilities for each sample.

        Args:
            X: Input tensor ``(N, seq_len, n_features)``.

        Returns:
            Array of shape ``(N, NUM_CLASSES)`` with softmax probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")

        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # ---- persistence -------------------------------------------------------
    def save_model(self, path: str | Path) -> None:
        """Persist the model, scaler, and metadata to *path*."""
        if self.model is None:
            raise RuntimeError("No model to save -- train or load one first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "input_size": self.model.rnn.input_size,
            "scaler": self.scaler,
        }
        torch.save(checkpoint, path)
        logger.info("Model saved to %s", path)

    def load_model(self, path: str | Path) -> None:
        """Load a previously saved model checkpoint from *path*."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model_type = checkpoint["model_type"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.scaler = checkpoint.get("scaler")

        bidirectional = self.model_type == "bilstm"
        cell_type = "gru" if self.model_type == "gru" else "lstm"

        self.model = LSTMPredictor(
            input_size=checkpoint["input_size"],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=bidirectional,
            cell_type=cell_type,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info("Model loaded from %s (type=%s).", path, self.model_type)
