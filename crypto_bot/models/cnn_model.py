"""
CNN model for candlestick chart pattern recognition.

Generates fixed-size RGB images from OHLCV data and trains a small
convolutional neural network to classify price patterns into BUY, HOLD,
or SELL signals.

Usage::

    from crypto_bot.config.settings import Settings
    from crypto_bot.models.cnn_model import (
        CandlestickImageGenerator,
        CNNTrainer,
    )

    gen = CandlestickImageGenerator(image_size=(64, 64))
    images, labels = gen.generate_dataset(ohlcv_df, window=20)

    trainer = CNNTrainer(Settings())
    metrics = trainer.train(images, labels, epochs=50)
    predictions = trainer.predict(images)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)


# ======================================================================
# Image generation
# ======================================================================


class CandlestickImageGenerator:
    """Generate fixed-size RGB candlestick chart images from OHLCV data.

    Parameters
    ----------
    image_size:
        ``(height, width)`` of the output images in pixels.
    """

    def __init__(self, image_size: Tuple[int, int] = (64, 64)) -> None:
        self.image_height, self.image_width = image_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_candlestick_image(
        self,
        ohlcv_data: pd.DataFrame,
        window: int = 20,
    ) -> np.ndarray:
        """Render the last *window* candles of *ohlcv_data* as an RGB image.

        Parameters
        ----------
        ohlcv_data:
            DataFrame with columns ``open``, ``high``, ``low``, ``close``
            (case-insensitive).  At least *window* rows are required.
        window:
            Number of candles to draw.

        Returns
        -------
        np.ndarray
            ``uint8`` array of shape ``(H, W, 3)``.
        """
        df = self._normalise_columns(ohlcv_data)
        df = df.iloc[-window:]

        if len(df) < window:
            raise ValueError(
                f"Need at least {window} rows; got {len(df)}"
            )

        return self._render(df)

    def generate_multi_resolution(
        self,
        ohlcv_data: pd.DataFrame,
        windows: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Stack multi-resolution candlestick images along the channel axis.

        Parameters
        ----------
        ohlcv_data:
            OHLCV DataFrame.
        windows:
            List of lookback windows (default ``[10, 20, 60]``).

        Returns
        -------
        np.ndarray
            ``uint8`` array of shape ``(H, W, 3 * len(windows))``.
        """
        if windows is None:
            windows = [10, 20, 60]

        images: List[np.ndarray] = []
        for w in windows:
            if len(ohlcv_data) >= w:
                img = self.generate_candlestick_image(ohlcv_data, window=w)
            else:
                # Pad with a blank image when insufficient data.
                img = np.zeros(
                    (self.image_height, self.image_width, 3), dtype=np.uint8,
                )
            images.append(img)

        return np.concatenate(images, axis=2)

    def generate_dataset(
        self,
        ohlcv_data: pd.DataFrame,
        window: int = 20,
        step: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create an image dataset with labels from a contiguous OHLCV series.

        Labels are derived from next-period returns:
            * BUY  (``1``)  if return > +0.5 %
            * SELL (``-1``) if return < -0.5 %
            * HOLD (``0``)  otherwise

        Parameters
        ----------
        ohlcv_data:
            OHLCV DataFrame.
        window:
            Number of candles per image.
        step:
            Stride between consecutive windows.

        Returns
        -------
        (images, labels)
            *images* has shape ``(N, H, W, 3)`` (``uint8``);
            *labels* has shape ``(N,)`` with values in ``{-1, 0, 1}``.
        """
        df = self._normalise_columns(ohlcv_data).reset_index(drop=True)

        images: List[np.ndarray] = []
        labels: List[int] = []

        # We need at least `window` candles for the image and one more row
        # after the window for the return-based label.
        max_start = len(df) - window - 1

        for start in range(0, max_start + 1, step):
            end = start + window
            chunk = df.iloc[start:end]

            try:
                img = self._render(chunk)
            except Exception:
                continue

            # Label from the return of the candle immediately after the window.
            close_current = float(df.iloc[end - 1]["close"])
            close_next = float(df.iloc[end]["close"])

            if close_current == 0:
                continue
            ret = (close_next - close_current) / close_current

            if ret > 0.005:
                label = 1   # BUY
            elif ret < -0.005:
                label = -1  # SELL
            else:
                label = 0   # HOLD

            images.append(img)
            labels.append(label)

        if not images:
            return (
                np.empty((0, self.image_height, self.image_width, 3), dtype=np.uint8),
                np.empty((0,), dtype=int),
            )

        return np.stack(images), np.array(labels, dtype=int)

    # ------------------------------------------------------------------
    # Rendering internals
    # ------------------------------------------------------------------

    def _render(self, df: pd.DataFrame) -> np.ndarray:
        """Render a normalised OHLCV slice as an RGB image."""
        img = np.zeros(
            (self.image_height, self.image_width, 3), dtype=np.uint8,
        )

        n_candles = len(df)
        candle_width = max(1, self.image_width // (n_candles * 2))
        spacing = self.image_width / n_candles

        # Price range for y-axis mapping.
        price_min = float(df["low"].min())
        price_max = float(df["high"].max())
        price_range = price_max - price_min
        if price_range == 0:
            price_range = 1.0  # Avoid division by zero for flat data.

        for i, (_, row) in enumerate(df.iterrows()):
            x = int(spacing * i + spacing / 2)
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])  # noqa: E741
            c = float(row["close"])

            self._draw_candle(img, x, o, h, l, c, candle_width, price_min, price_range)

        return img

    def _draw_candle(
        self,
        img: np.ndarray,
        x: int,
        open_price: float,
        high: float,
        low: float,
        close: float,
        width: int,
        price_min: float,
        price_range: float,
    ) -> None:
        """Draw a single candlestick on the image array.

        Parameters
        ----------
        img:
            Mutable ``(H, W, 3)`` uint8 array.
        x:
            Horizontal centre pixel.
        open_price, high, low, close:
            OHLC prices.
        width:
            Half-width of the candle body in pixels.
        price_min, price_range:
            Used to map prices to pixel rows.
        """
        h = self.image_height

        def _price_to_y(price: float) -> int:
            """Map a price value to a pixel row (top of image = high price)."""
            normalised = (price - price_min) / price_range
            return int(h - 1 - normalised * (h - 1))

        y_open = _price_to_y(open_price)
        y_close = _price_to_y(close)
        y_high = _price_to_y(high)
        y_low = _price_to_y(low)

        bullish = close >= open_price
        # Green for bullish, red for bearish.
        colour = (0, 200, 0) if bullish else (200, 0, 0)

        # Draw the wick (high-low line).
        wick_top = min(y_high, y_low)
        wick_bot = max(y_high, y_low)
        x_clamped = np.clip(x, 0, self.image_width - 1)
        for row in range(max(0, wick_top), min(h, wick_bot + 1)):
            img[row, x_clamped] = colour

        # Draw the body (open-close rectangle).
        body_top = min(y_open, y_close)
        body_bot = max(y_open, y_close)
        # Ensure at least 1-pixel body.
        if body_top == body_bot:
            body_bot = body_top + 1

        x_left = max(0, x - width)
        x_right = min(self.image_width, x + width + 1)

        for row in range(max(0, body_top), min(h, body_bot + 1)):
            img[row, x_left:x_right] = colour

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure lowercase column names and required OHLC columns exist."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df


# ======================================================================
# CNN architecture
# ======================================================================


class CandlestickCNN(nn.Module):
    """Small convolutional neural network for candlestick pattern recognition.

    Architecture::

        Conv2d(in, 32)  -> ReLU -> MaxPool
        Conv2d(32, 64)  -> ReLU -> MaxPool
        Conv2d(64, 128) -> ReLU -> AdaptiveAvgPool(4,4)
        Flatten -> FC(512) -> ReLU -> Dropout(0.5) -> FC(num_classes)

    Parameters
    ----------
    num_classes:
        Number of output classes (default 3: BUY, HOLD, SELL).
    in_channels:
        Number of input image channels (default 3 for RGB).
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_classes)``.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


# ======================================================================
# Trainer wrapper
# ======================================================================


class CNNTrainer:
    """High-level trainer for :class:`CandlestickCNN`.

    Manages device placement, training loop, prediction, and model
    persistence.

    Parameters
    ----------
    settings:
        Project-wide settings.
    num_classes:
        Number of output classes (default 3).
    in_channels:
        Number of input image channels (default 3).
    """

    def __init__(
        self,
        settings: Settings,
        num_classes: int = 3,
        in_channels: int = 3,
    ) -> None:
        self._settings = settings
        self._num_classes = num_classes
        self._in_channels = in_channels

        # Device selection.
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        self._model = CandlestickCNN(
            num_classes=num_classes,
            in_channels=in_channels,
        ).to(self._device)

        self._is_trained: bool = False

        logger.info(
            "CNNTrainer initialised on device '%s' (classes=%d, channels=%d)",
            self._device,
            num_classes,
            in_channels,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        val_split: float = 0.2,
    ) -> Dict[str, Any]:
        """Train the CNN on candlestick images.

        Parameters
        ----------
        images:
            ``uint8`` array of shape ``(N, H, W, C)`` (channels-last).
        labels:
            Integer labels ``{-1, 0, 1}`` of shape ``(N,)``.
        epochs:
            Number of training epochs.
        batch_size:
            Mini-batch size.
        learning_rate:
            Adam optimiser learning rate.
        val_split:
            Fraction of data used for validation.

        Returns
        -------
        dict
            Keys: ``train_loss``, ``val_loss``, ``train_accuracy``,
            ``val_accuracy``, ``epochs_trained``.
        """
        # Remap labels: {-1, 0, 1} -> {0, 1, 2}.
        labels_mapped = np.asarray(labels, dtype=np.int64) + 1

        # Convert images to float tensors in [0, 1], channels-first.
        img_tensor = (
            torch.from_numpy(images)
            .permute(0, 3, 1, 2)
            .float()
            / 255.0
        )
        lbl_tensor = torch.from_numpy(labels_mapped).long()

        # Train / val split.
        n_total = len(img_tensor)
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val

        indices = torch.randperm(n_total)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        train_ds = TensorDataset(img_tensor[train_idx], lbl_tensor[train_idx])
        val_ds = TensorDataset(img_tensor[val_idx], lbl_tensor[val_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        optimiser = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state: Optional[Dict[str, Any]] = None

        logger.info(
            "Training CNN: %d train / %d val samples, %d epochs",
            n_train,
            n_val,
            epochs,
        )

        for epoch in range(1, epochs + 1):
            # --- Training phase ---
            self._model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for batch_imgs, batch_lbls in train_loader:
                batch_imgs = batch_imgs.to(self._device)
                batch_lbls = batch_lbls.to(self._device)

                optimiser.zero_grad()
                logits = self._model(batch_imgs)
                loss = criterion(logits, batch_lbls)
                loss.backward()
                optimiser.step()

                train_loss_sum += loss.item() * batch_imgs.size(0)
                preds = logits.argmax(dim=1)
                train_correct += (preds == batch_lbls).sum().item()
                train_total += batch_imgs.size(0)

            train_loss = train_loss_sum / max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            # --- Validation phase ---
            self._model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_imgs, batch_lbls in val_loader:
                    batch_imgs = batch_imgs.to(self._device)
                    batch_lbls = batch_lbls.to(self._device)

                    logits = self._model(batch_imgs)
                    loss = criterion(logits, batch_lbls)

                    val_loss_sum += loss.item() * batch_imgs.size(0)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == batch_lbls).sum().item()
                    val_total += batch_imgs.size(0)

            val_loss = val_loss_sum / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %d/%d -- train_loss=%.4f, val_loss=%.4f, "
                    "train_acc=%.4f, val_acc=%.4f",
                    epoch,
                    epochs,
                    train_loss,
                    val_loss,
                    train_acc,
                    val_acc,
                )

        # Restore the best model weights.
        if best_state is not None:
            self._model.load_state_dict(best_state)
            self._model.to(self._device)

        self._is_trained = True

        metrics: Dict[str, Any] = {
            "train_loss": round(train_loss, 4),
            "val_loss": round(best_val_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "epochs_trained": epochs,
        }

        logger.info("CNN training complete: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Return trading signal predictions for a batch of images.

        Parameters
        ----------
        images:
            ``uint8`` array of shape ``(N, H, W, C)`` or ``(H, W, C)``.

        Returns
        -------
        np.ndarray
            Integer predictions in ``{-1, 0, 1}`` (SELL / HOLD / BUY).
        """
        self._check_trained()
        probs = self.predict_proba(images)
        internal_preds = np.argmax(probs, axis=1)
        return internal_preds - 1  # {0,1,2} -> {-1,0,1}

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """Return class probabilities for a batch of images.

        Parameters
        ----------
        images:
            ``uint8`` array of shape ``(N, H, W, C)`` or ``(H, W, C)``.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, 3)`` with columns ``[SELL, HOLD, BUY]``.
        """
        self._check_trained()

        if images.ndim == 3:
            images = images[np.newaxis, ...]

        img_tensor = (
            torch.from_numpy(images)
            .permute(0, 3, 1, 2)
            .float()
            / 255.0
        ).to(self._device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(img_tensor)
            probs = F.softmax(logits, dim=1)

        return probs.cpu().numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: Union[str, Path]) -> None:
        """Save model weights to disk.

        Parameters
        ----------
        path:
            Destination file path (typically ``*.pt`` or ``*.pth``).
        """
        self._check_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "num_classes": self._num_classes,
                "in_channels": self._in_channels,
            },
            str(path),
        )
        logger.info("CNN model saved to %s", path)

    def load_model(self, path: Union[str, Path]) -> None:
        """Load model weights from disk.

        Parameters
        ----------
        path:
            Path to a ``.pt`` / ``.pth`` checkpoint.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

        checkpoint = torch.load(str(path), map_location=self._device, weights_only=False)

        num_classes = checkpoint.get("num_classes", self._num_classes)
        in_channels = checkpoint.get("in_channels", self._in_channels)

        # Rebuild the model if architecture parameters differ.
        if num_classes != self._num_classes or in_channels != self._in_channels:
            self._num_classes = num_classes
            self._in_channels = in_channels
            self._model = CandlestickCNN(
                num_classes=num_classes,
                in_channels=in_channels,
            ).to(self._device)

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self._device)
        self._is_trained = True
        logger.info("CNN model loaded from %s", path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        """Raise if the model has not been trained or loaded."""
        if not self._is_trained:
            raise RuntimeError(
                "CNNTrainer model is not trained. "
                "Call train() or load_model() first."
            )
