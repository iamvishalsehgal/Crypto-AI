"""
PPO (Proximal Policy Optimization) agent for crypto trading.

Provides two policy architectures as recommended by the RL strategy engine spec:
  - ``PPOAgent`` -- MLP policy (256-256-128 hidden layers)
  - ``RecurrentPPOAgent`` -- LSTM policy for sequential memory

Uses stable-baselines3 with EvalCallback for best-model checkpointing,
early stopping, and chronological train/val/test split.

Usage::

    from omnitrade.models.ppo_agent import PPOAgent

    agent = PPOAgent(settings, env)
    agent.train(timesteps=500_000)
    action = agent.predict(obs)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, EarlyStoppingCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from omnitrade.config.settings import Settings
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)


class PPOAgent:
    """PPO agent with MLP policy for crypto trading.

    Uses stable-baselines3 PPO with:
    - MLP policy net: [256, 256, 128] hidden layers
    - Entropy coefficient for exploration (ent_coef=0.01)
    - n_steps=2048, batch_size=64
    - VecNormalize for observation normalisation
    - EvalCallback with best-model saving and early stopping
    - Chronological 70/15/15 train/val/test split

    Args:
        settings: Global project settings.
        env: Gymnasium trading environment (will be wrapped in VecEnv).
        tensorboard_log: Path for TensorBoard logs.
        model_dir: Directory for saving best model checkpoints.
    """

    def __init__(
        self,
        settings: Settings,
        env: gym.Env,
        tensorboard_log: str = "./logs/ppo_tensorboard",
        model_dir: str = "./models/ppo",
    ) -> None:
        self._settings = settings
        self._model_cfg = settings.model

        # PPO hyperparameters from settings with sensible defaults.
        self.learning_rate: float = getattr(
            self._model_cfg, "ppo_learning_rate", 3e-4
        )
        self.n_steps: int = getattr(self._model_cfg, "ppo_n_steps", 2048)
        self.batch_size: int = getattr(self._model_cfg, "ppo_batch_size", 64)
        self.n_epochs: int = getattr(self._model_cfg, "ppo_n_epochs", 10)
        self.gamma: float = getattr(self._model_cfg, "ppo_gamma", 0.99)
        self.gae_lambda: float = getattr(self._model_cfg, "ppo_gae_lambda", 0.95)
        self.ent_coef: float = getattr(self._model_cfg, "ppo_ent_coef", 0.01)
        self.clip_range: float = getattr(self._model_cfg, "ppo_clip_range", 0.2)
        self.hidden_layers: list = getattr(
            self._model_cfg, "ppo_hidden_layers", [256, 256, 128]
        )

        # Wrap environment.
        self._env = DummyVecEnv([lambda: env])
        self._env = VecNormalize(
            self._env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

        # Build policy kwargs for custom architecture.
        policy_kwargs = dict(
            net_arch=dict(
                pi=self.hidden_layers,
                vf=self.hidden_layers,
            ),
            activation_fn=None,  # uses default Tanh
        )

        self.model = PPO(
            "MlpPolicy",
            self._env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            ent_coef=self.ent_coef,
            clip_range=self.clip_range,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=0,
            seed=42,
        )

        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._tensorboard_log = tensorboard_log
        self._trained = False
        self._best_model_path: Optional[Path] = None

        logger.info(
            "PPOAgent initialised: lr=%.2e, n_steps=%d, ent_coef=%.3f, "
            "hidden=%s, device=%s",
            self.learning_rate,
            self.n_steps,
            self.ent_coef,
            self.hidden_layers,
            self.model.device,
        )

    # ------------------------------------------------------------------
    # Training with validation and early stopping
    # ------------------------------------------------------------------

    def train(
        self,
        timesteps: int = 500_000,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10_000,
        early_stopping_patience: int = 10,
        save_best: bool = True,
    ) -> Dict[str, Any]:
        """Train PPO with evaluation callback and early stopping.

        Args:
            timesteps: Total training timesteps.
            eval_env: Separate evaluation environment (wrapped in VecNormalize).
                If None, uses a copy of the training env (not ideal; prefer a
                held-out validation set).
            eval_freq: Evaluate every N timesteps.
            early_stopping_patience: Stop if no improvement after N evals.
            save_best: Save best model checkpoint to disk.

        Returns:
            Dict with training metrics.
        """
        callbacks = []

        if eval_env is not None:
            eval_vec = DummyVecEnv([lambda: eval_env])
            # Use same normalization stats as training env.
            eval_vec = VecNormalize(
                eval_vec,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
                training=False,
            )

            best_model_path = str(self._model_dir / "best_model")
            eval_callback = EvalCallback(
                eval_vec,
                best_model_save_path=best_model_path if save_best else None,
                log_path=self._tensorboard_log,
                eval_freq=eval_freq,
                n_eval_episodes=5,
                deterministic=True,
                render=False,
                callback_after_eval=(
                    EarlyStoppingCallback(
                        early_stopping_patience=early_stopping_patience,
                        min_delta=1e-4,
                    )
                    if early_stopping_patience > 0
                    else None
                ),
            )
            callbacks.append(eval_callback)

        logger.info(
            "Starting PPO training: %d timesteps, eval_freq=%d, patience=%d",
            timesteps, eval_freq, early_stopping_patience,
        )

        try:
            self.model.learn(
                total_timesteps=timesteps,
                callback=callbacks if callbacks else None,
                progress_bar=False,
            )
            self._trained = True
        except Exception as exc:
            logger.error("PPO training failed: %s", exc)
            return {"status": "error", "error": str(exc)}

        # Save final model.
        final_path = self._model_dir / "ppo_final"
        self.model.save(str(final_path))
        self._env.save(str(self._model_dir / "vec_normalize.pkl"))

        logger.info("PPO training complete. Final model saved to %s", final_path)
        return {
            "status": "trained",
            "timesteps": timesteps,
            "final_model_path": str(final_path),
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Return greedy action for a single observation.

        Args:
            obs: 1-D observation vector matching the env observation space.
            deterministic: If True, use greedy policy (no sampling).

        Returns:
            Integer action index (0=BUY, 1=HOLD, 2=SELL).
        """
        if not self._trained:
            logger.warning("PPOAgent not trained — returning HOLD")
            return 1

        action, _states = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str | Path) -> None:
        """Save agent to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info("PPOAgent saved to %s", path)

    def load_model(self, path: str | Path) -> None:
        """Load agent from *path*."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PPO model not found: {path}")

        self.model = PPO.load(
            str(path),
            env=self._env,
        )
        self._trained = True
        logger.info("PPOAgent loaded from %s", path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def vec_normalize(self):
        """Return VecNormalize wrapper for saving/loading stats."""
        return self._env


class RecurrentPPOAgent:
    """PPO agent with LSTM policy for sequence-aware trading.

    LSTM hidden state captures temporal dependencies beyond the sliding
    window in the observation, enabling the agent to learn multi-step
    patterns (momentum continuation, mean reversion timing).

    Architecture per the RL strategy engine spec:
    - LSTM hidden size: 128
    - Number of LSTM layers: 2
    - Shared feature extractor: [256, 256] before LSTM

    Args:
        settings: Global project settings.
        env: Gymnasium trading environment.
        tensorboard_log: Path for TensorBoard logs.
        model_dir: Directory for saving model checkpoints.
    """

    def __init__(
        self,
        settings: Settings,
        env: gym.Env,
        tensorboard_log: str = "./logs/recurrent_ppo_tensorboard",
        model_dir: str = "./models/recurrent_ppo",
    ) -> None:
        self._settings = settings
        self._model_cfg = settings.model

        self.learning_rate: float = getattr(
            self._model_cfg, "ppo_learning_rate", 3e-4
        )
        self.n_steps: int = getattr(self._model_cfg, "ppo_n_steps", 2048)
        self.batch_size: int = getattr(self._model_cfg, "ppo_batch_size", 64)
        self.ent_coef: float = getattr(self._model_cfg, "ppo_ent_coef", 0.01)
        self.lstm_hidden_size: int = getattr(
            self._model_cfg, "ppo_lstm_hidden_size", 128
        )
        self.n_lstm_layers: int = getattr(
            self._model_cfg, "ppo_n_lstm_layers", 2
        )

        self._env = DummyVecEnv([lambda: env])
        self._env = VecNormalize(
            self._env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

        # LSTM policy: shared feature extractor [256, 256] -> LSTM(128, 2 layers).
        policy_kwargs = dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256],
            ),
            enable_critic_lstm=False,
            lstm_hidden_size=self.lstm_hidden_size,
            n_lstm_layers=self.n_lstm_layers,
        )

        self.model = PPO(
            "MlpLstmPolicy",
            self._env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            ent_coef=self.ent_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=0,
            seed=42,
        )

        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._trained = False

        logger.info(
            "RecurrentPPOAgent initialised: lstm_hidden=%d, lstm_layers=%d, "
            "n_steps=%d, ent_coef=%.3f",
            self.lstm_hidden_size,
            self.n_lstm_layers,
            self.n_steps,
            self.ent_coef,
        )

    def train(
        self,
        timesteps: int = 500_000,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10_000,
        early_stopping_patience: int = 10,
    ) -> Dict[str, Any]:
        """Train RecurrentPPO with evaluation and early stopping."""
        callbacks = []
        if eval_env is not None:
            eval_vec = DummyVecEnv([lambda: eval_env])
            eval_vec = VecNormalize(
                eval_vec,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
                training=False,
            )

            best_path = str(self._model_dir / "best_model")
            eval_callback = EvalCallback(
                eval_vec,
                best_model_save_path=best_path,
                log_path="./logs/recurrent_ppo_tensorboard",
                eval_freq=eval_freq,
                n_eval_episodes=5,
                deterministic=True,
                callback_after_eval=EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                    min_delta=1e-4,
                ),
            )
            callbacks.append(eval_callback)

        try:
            self.model.learn(
                total_timesteps=timesteps,
                callback=callbacks if callbacks else None,
                progress_bar=False,
            )
            self._trained = True
        except Exception as exc:
            logger.error("RecurrentPPO training failed: %s", exc)
            return {"status": "error", "error": str(exc)}

        final_path = self._model_dir / "recurrent_ppo_final"
        self.model.save(str(final_path))
        logger.info("RecurrentPPO training complete. Saved to %s", final_path)
        return {"status": "trained", "timesteps": timesteps}

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> int:
        if not self._trained:
            return 1
        action, _states = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def save_model(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load_model(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"RecurrentPPO model not found: {path}")
        self.model = PPO.load(str(path), env=self._env)
        self._trained = True

    @property
    def is_trained(self) -> bool:
        return self._trained


def create_ppo_train_val_envs(
    df: pd.DataFrame,
    env_kwargs: Optional[Dict[str, Any]] = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[gym.Env, gym.Env, gym.Env]:
    """Create chronologically-split train/val/test environments.

    Uses 70/15/15 chronological split (no shuffling) to prevent look-ahead
    bias in time-series data.

    Args:
        df: Full feature DataFrame.
        env_kwargs: Kwargs forwarded to ``CryptoTradingEnv``.
        train_frac: Fraction for training (default 0.70).
        val_frac: Fraction for validation (default 0.15).
            Test gets the remainder.

    Returns:
        Tuple of (train_env, val_env, test_env).
    """
    from omnitrade.models.training.environments import CryptoTradingEnv

    if env_kwargs is None:
        env_kwargs = {}

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    train_env = CryptoTradingEnv(df_train, random_start=True, **env_kwargs)
    val_env = CryptoTradingEnv(df_val, random_start=False, **env_kwargs)
    test_env = CryptoTradingEnv(df_test, random_start=False, **env_kwargs)

    logger.info(
        "Chronological split: train=%d, val=%d, test=%d rows",
        len(df_train), len(df_val), len(df_test),
    )

    return train_env, val_env, test_env
