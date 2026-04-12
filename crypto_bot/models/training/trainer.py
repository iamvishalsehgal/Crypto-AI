"""
Training orchestrator for the AI Crypto Trading Bot.

Coordinates data splitting (time-series aware), model training (LSTM / BiLSTM
/ GRU / DDQN), evaluation, and comparison.  All splits are chronological to
prevent future-data leakage.

Usage::

    from crypto_bot.models.training.trainer import ModelTrainer

    trainer = ModelTrainer(settings)
    results = trainer.train_all_models(features_df)
    comparison = trainer.compare_models(results)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from crypto_bot.config.settings import Settings
from crypto_bot.models.ddqn_agent import DDQNAgent
from crypto_bot.models.lstm_model import LSTMTrainer
from crypto_bot.models.training.environments import CryptoTradingEnv
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """High-level training orchestrator.

    Provides helpers for chronological data splitting, training each model
    type, evaluation, and cross-model comparison.

    Args:
        settings: Global project settings.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("ModelTrainer initialised (device=%s).", self.device)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prepare_data(
        self,
        features_df: pd.DataFrame,
        target_col: str = "signal",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split features into chronological train / val / test partitions.

        The split is purely time-ordered -- no shuffling -- to prevent future
        data leakage.

        Args:
            features_df: DataFrame containing feature columns and
                *target_col*.
            target_col: Name of the target column.
            train_ratio: Fraction of data for training.
            val_ratio: Fraction of data for validation.

        Returns:
            ``(X_train, X_val, X_test, y_train, y_val, y_test)``
        """
        if target_col not in features_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        n = len(features_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        feature_cols = [c for c in features_df.columns if c != target_col]

        X_train = features_df.iloc[:train_end][feature_cols]
        X_val = features_df.iloc[train_end:val_end][feature_cols]
        X_test = features_df.iloc[val_end:][feature_cols]

        y_train = features_df.iloc[:train_end][target_col]
        y_val = features_df.iloc[train_end:val_end][target_col]
        y_test = features_df.iloc[val_end:][target_col]

        logger.info(
            "Data split: train=%d, val=%d, test=%d (features=%d)",
            len(X_train),
            len(X_val),
            len(X_test),
            len(feature_cols),
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------
    # LSTM training
    # ------------------------------------------------------------------
    def train_lstm(
        self,
        features_df: pd.DataFrame,
        model_type: str = "lstm",
        target_col: str = "signal",
        sequence_length: int = 60,
        epochs: int = 100,
        patience: int = 10,
    ) -> Tuple[LSTMTrainer, Dict[str, Any]]:
        """Train an LSTM-family model with walk-forward validation.

        Args:
            features_df: DataFrame with feature columns and a *target_col*.
            model_type: One of ``'lstm'``, ``'bilstm'``, ``'gru'``.
            target_col: Name of the target column.
            sequence_length: Sliding-window length for sequences.
            epochs: Maximum training epochs.
            patience: Early-stopping patience.

        Returns:
            ``(trainer, metrics)`` where *metrics* is a dictionary of
            aggregated walk-forward results.
        """
        logger.info("Training %s model with walk-forward validation.", model_type.upper())

        n_windows: int = getattr(
            self.settings.backtesting, "walk_forward_windows", 5
        )
        n = len(features_df)

        # Minimum window size to have enough data for sequences + validation.
        min_window = sequence_length * 3
        if n < min_window * 2:
            logger.warning(
                "Not enough data (%d rows) for walk-forward with %d windows. "
                "Falling back to single split.",
                n,
                n_windows,
            )
            n_windows = 1

        fold_metrics: List[Dict[str, float]] = []
        best_trainer: Optional[LSTMTrainer] = None
        best_val_loss = float("inf")

        # Walk-forward: expanding or rolling window approach.
        segment_size = n // (n_windows + 1)

        for fold in range(n_windows):
            train_end = segment_size * (fold + 1)
            val_end = min(train_end + segment_size, n)

            if train_end < min_window or val_end - train_end < sequence_length:
                logger.debug("Skipping fold %d -- not enough data.", fold)
                continue

            fold_df = features_df.iloc[:val_end].copy()

            trainer = LSTMTrainer(self.settings, model_type=model_type)
            X_train, y_train, X_val, y_val = trainer.prepare_sequences(
                fold_df.iloc[:train_end],
                sequence_length=sequence_length,
                target_col=target_col,
                val_ratio=0.15,
            )

            # Also prepare an out-of-sample validation set from the next segment.
            oos_trainer = LSTMTrainer(self.settings, model_type=model_type)
            oos_trainer.scaler = trainer.scaler  # reuse training scaler
            oos_data = features_df.iloc[train_end:val_end]
            if len(oos_data) > sequence_length:
                oos_feature_cols = [c for c in oos_data.columns if c != target_col]
                oos_features = trainer.scaler.transform(
                    oos_data[oos_feature_cols].values.astype(np.float32)
                )
                oos_targets = oos_data[target_col].values.astype(np.int64)

                X_oos_list = []
                y_oos_list = []
                for i in range(sequence_length, len(oos_features)):
                    X_oos_list.append(oos_features[i - sequence_length : i])
                    y_oos_list.append(oos_targets[i])

                if X_oos_list:
                    X_oos = torch.tensor(np.array(X_oos_list), dtype=torch.float32)
                    y_oos = torch.tensor(np.array(y_oos_list), dtype=torch.long)
                else:
                    X_oos, y_oos = X_val, y_val
            else:
                X_oos, y_oos = X_val, y_val

            history = trainer.train(
                X_train, y_train, X_val, y_val, epochs=epochs, patience=patience
            )

            # Evaluate on out-of-sample data.
            preds_proba = trainer.predict(X_oos)
            preds = preds_proba.argmax(axis=1)
            y_true = y_oos.numpy()

            fold_result = {
                "fold": fold,
                "accuracy": float(accuracy_score(y_true, preds)),
                "precision": float(precision_score(y_true, preds, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_true, preds, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
                "val_loss": float(min(history["val_loss"])) if history["val_loss"] else float("inf"),
            }
            fold_metrics.append(fold_result)

            if fold_result["val_loss"] < best_val_loss:
                best_val_loss = fold_result["val_loss"]
                best_trainer = trainer

            logger.info(
                "Fold %d/%d  acc=%.3f  f1=%.3f  val_loss=%.4f",
                fold + 1,
                n_windows,
                fold_result["accuracy"],
                fold_result["f1"],
                fold_result["val_loss"],
            )

        # Aggregate metrics across folds.
        if not fold_metrics:
            raise RuntimeError("No folds completed -- insufficient data.")

        aggregated: Dict[str, Any] = {
            "model_type": model_type,
            "n_folds": len(fold_metrics),
            "mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
            "mean_precision": float(np.mean([m["precision"] for m in fold_metrics])),
            "mean_recall": float(np.mean([m["recall"] for m in fold_metrics])),
            "mean_f1": float(np.mean([m["f1"] for m in fold_metrics])),
            "std_f1": float(np.std([m["f1"] for m in fold_metrics])),
            "best_val_loss": best_val_loss,
            "fold_details": fold_metrics,
        }

        logger.info(
            "%s walk-forward complete: mean_f1=%.3f (+-%.3f), mean_acc=%.3f",
            model_type.upper(),
            aggregated["mean_f1"],
            aggregated["std_f1"],
            aggregated["mean_accuracy"],
        )

        if best_trainer is None:
            raise RuntimeError("No valid trainer produced across folds.")

        return best_trainer, aggregated

    # ------------------------------------------------------------------
    # DDQN training
    # ------------------------------------------------------------------
    def train_ddqn(
        self,
        features_df: pd.DataFrame,
        episodes: int = 1000,
        max_steps: Optional[int] = None,
        initial_balance: float = 10_000.0,
    ) -> Tuple[DDQNAgent, Dict[str, Any]]:
        """Train a DDQN agent on the features DataFrame.

        Constructs a :class:`CryptoTradingEnv` internally and runs the
        agent's training loop.

        Args:
            features_df: DataFrame containing OHLCV / feature columns (must
                have a ``close`` column).
            episodes: Number of RL training episodes.
            max_steps: Per-episode step cap.
            initial_balance: Starting balance for the environment.

        Returns:
            ``(agent, metrics)``
        """
        logger.info("Training DDQN agent for %d episodes.", episodes)

        # Split data: 80 % train env, 20 % eval env.
        split_idx = int(len(features_df) * 0.8)
        train_df = features_df.iloc[:split_idx].reset_index(drop=True)
        eval_df = features_df.iloc[split_idx:].reset_index(drop=True)

        fee = getattr(self.settings.backtesting, "transaction_fee", 0.001)
        slip = getattr(self.settings.backtesting, "slippage", 0.0005)

        train_env = CryptoTradingEnv(
            train_df,
            initial_balance=initial_balance,
            transaction_fee=fee,
            slippage=slip,
        )

        state_size = train_env.observation_space.shape[0]
        agent = DDQNAgent(self.settings, state_size=state_size, action_size=3)
        history = agent.train(train_env, episodes=episodes, max_steps=max_steps)

        # Evaluate on unseen data.
        eval_env = CryptoTradingEnv(
            eval_df,
            initial_balance=initial_balance,
            transaction_fee=fee,
            slippage=slip,
        )

        eval_rewards, eval_trades, eval_equities = self._evaluate_ddqn(agent, eval_env)

        # Compute Sharpe-like ratio from episode returns.
        if len(eval_rewards) > 1:
            returns = np.array(eval_rewards)
            sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
        else:
            sharpe = 0.0

        metrics: Dict[str, Any] = {
            "model_type": "ddqn",
            "training_episodes": episodes,
            "mean_train_reward": float(np.mean(history["episode_rewards"])),
            "best_train_reward": float(np.max(history["episode_rewards"])),
            "mean_eval_reward": float(np.mean(eval_rewards)) if eval_rewards else 0.0,
            "eval_sharpe_ratio": sharpe,
            "eval_total_trades": sum(eval_trades),
            "final_epsilon": agent.epsilon,
            "training_history": history,
        }

        logger.info(
            "DDQN training complete: mean_eval_reward=%.2f, sharpe=%.3f, trades=%d",
            metrics["mean_eval_reward"],
            sharpe,
            metrics["eval_total_trades"],
        )

        return agent, metrics

    # ------------------------------------------------------------------
    # Train all models
    # ------------------------------------------------------------------
    def train_all_models(
        self,
        features_df: pd.DataFrame,
        target_col: str = "signal",
    ) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        """Train every supported model and return results.

        Trains: LSTM, BiLSTM, GRU, DDQN.

        Args:
            features_df: DataFrame with features and *target_col*.
            target_col: Target column name (used by LSTM-family models).

        Returns:
            ``{model_name: (model_object, metrics_dict)}``
        """
        results: Dict[str, Tuple[Any, Dict[str, Any]]] = {}

        for model_type in ("lstm", "bilstm", "gru"):
            try:
                logger.info("--- Training %s ---", model_type.upper())
                model, metrics = self.train_lstm(
                    features_df, model_type=model_type, target_col=target_col
                )
                results[model_type] = (model, metrics)
            except Exception:
                logger.exception("Failed to train %s model.", model_type)

        try:
            logger.info("--- Training DDQN ---")
            agent, metrics = self.train_ddqn(features_df)
            results["ddqn"] = (agent, metrics)
        except Exception:
            logger.exception("Failed to train DDQN agent.")

        logger.info("All model training complete.  Trained: %s", list(results.keys()))
        return results

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_model(
        self,
        model: Any,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate a classification model on held-out test data.

        Works with :class:`LSTMTrainer` instances.

        Args:
            model: A trained ``LSTMTrainer`` (must expose ``.predict()``).
            X_test: Test feature tensor ``(N, seq_len, n_features)``.
            y_test: Test label tensor ``(N,)``.

        Returns:
            Dictionary with ``accuracy``, ``precision``, ``recall``, ``f1``,
            and ``sharpe_ratio``.
        """
        if isinstance(model, LSTMTrainer):
            probs = model.predict(X_test)
            preds = probs.argmax(axis=1)
        else:
            raise TypeError(f"Unsupported model type for evaluate_model: {type(model)}")

        y_true = y_test.numpy() if isinstance(y_test, torch.Tensor) else np.asarray(y_test)

        acc = float(accuracy_score(y_true, preds))
        prec = float(precision_score(y_true, preds, average="weighted", zero_division=0))
        rec = float(recall_score(y_true, preds, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))

        # Simple Sharpe proxy: treat each correct prediction of BUY (0) or
        # SELL (2) as +1 return, wrong as -1, HOLD as 0.
        pseudo_returns = []
        for pred, true in zip(preds, y_true):
            if pred == true and pred != 1:  # correct non-HOLD
                pseudo_returns.append(1.0)
            elif pred != true and pred != 1:  # wrong non-HOLD
                pseudo_returns.append(-1.0)
            else:
                pseudo_returns.append(0.0)

        returns_arr = np.array(pseudo_returns)
        sharpe = 0.0
        if len(returns_arr) > 1 and returns_arr.std() > 0:
            sharpe = float(returns_arr.mean() / returns_arr.std()) * np.sqrt(252)

        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "sharpe_ratio": sharpe,
        }

        logger.info(
            "Evaluation: acc=%.3f, prec=%.3f, rec=%.3f, f1=%.3f, sharpe=%.3f",
            acc,
            prec,
            rec,
            f1,
            sharpe,
        )
        return metrics

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------
    def compare_models(
        self, results: Dict[str, Tuple[Any, Dict[str, Any]]]
    ) -> pd.DataFrame:
        """Build a comparison table from training results.

        Args:
            results: Mapping of ``{model_name: (model, metrics_dict)}``.

        Returns:
            DataFrame with one row per model and key metric columns.
        """
        rows: List[Dict[str, Any]] = []
        for name, (_, metrics) in results.items():
            row: Dict[str, Any] = {"model": name}
            # Normalise field names across LSTM and DDQN metrics.
            row["accuracy"] = metrics.get("mean_accuracy", metrics.get("accuracy", None))
            row["precision"] = metrics.get("mean_precision", metrics.get("precision", None))
            row["recall"] = metrics.get("mean_recall", metrics.get("recall", None))
            row["f1"] = metrics.get("mean_f1", metrics.get("f1", None))
            row["sharpe_ratio"] = metrics.get("eval_sharpe_ratio", metrics.get("sharpe_ratio", None))
            row["val_loss"] = metrics.get("best_val_loss", None)
            rows.append(row)

        df = pd.DataFrame(rows).set_index("model")

        # Sort by F1 descending (where available).
        if "f1" in df.columns:
            df = df.sort_values("f1", ascending=False, na_position="last")

        logger.info("Model comparison:\n%s", df.to_string())
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evaluate_ddqn(
        self,
        agent: DDQNAgent,
        env: CryptoTradingEnv,
        n_episodes: int = 5,
    ) -> Tuple[List[float], List[int], List[float]]:
        """Run the agent greedily on *env* for *n_episodes* and collect stats.

        Returns:
            ``(rewards, trade_counts, final_equities)``
        """
        rewards: List[float] = []
        trade_counts: List[int] = []
        equities: List[float] = []

        for _ in range(n_episodes):
            obs, info = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = agent.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            rewards.append(total_reward)
            trade_counts.append(info.get("n_trades", 0))
            equities.append(info.get("equity", 0.0))

        return rewards, trade_counts, equities
