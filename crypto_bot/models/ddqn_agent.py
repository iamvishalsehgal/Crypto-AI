"""
Double Deep Q-Network (DDQN) agent for crypto trading.

Provides:
    - ``QNetwork`` -- feedforward PyTorch network that maps a state vector to
      Q-values for each discrete action.
    - ``DDQNAgent`` -- full training / inference agent with experience replay,
      epsilon-greedy exploration, and periodic target-network updates.

Usage::

    from crypto_bot.models.ddqn_agent import DDQNAgent

    agent = DDQNAgent(settings, state_size=env.observation_space.shape[0])
    history = agent.train(env, episodes=500)
    action = agent.predict(state)
"""

from __future__ import annotations

import copy
import random
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    """Feed-forward network that outputs Q-values for discrete actions.

    Architecture per hidden layer: ``Linear -> BatchNorm -> ReLU -> Dropout``.

    Args:
        state_size: Dimensionality of the state (observation) vector.
        action_size: Number of discrete actions.
        hidden_layers: Sequence of hidden-layer widths.
        dropout: Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128, 64]

        layers: List[nn.Module] = []
        prev_size = state_size
        for h_size in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev_size, h_size),
                    nn.BatchNorm1d(h_size),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
            )
            prev_size = h_size

        layers.append(nn.Linear(prev_size, action_size))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for each action.

        Args:
            state: Tensor of shape ``(batch, state_size)``.

        Returns:
            Tensor of shape ``(batch, action_size)``.
        """
        return self.network(state)


# ---------------------------------------------------------------------------
# Experience tuple
# ---------------------------------------------------------------------------
Experience = Tuple[np.ndarray, int, float, np.ndarray, bool]


# ---------------------------------------------------------------------------
# DDQN Agent
# ---------------------------------------------------------------------------
class DDQNAgent:
    """Double DQN agent for discrete-action crypto trading.

    The agent maintains two networks:
    * **policy_network** -- used to select actions and is updated every step.
    * **target_network** -- used to evaluate Q-values for the DDQN target and
      is periodically synchronised with the policy network.

    DDQN target:
        ``y = r + gamma * Q_target(s', argmax_a Q_policy(s', a))``

    Args:
        settings: Global project settings (hyper-params sourced from
            ``settings.model``).
        state_size: Dimensionality of the observation vector.
        action_size: Number of discrete actions (default 3: BUY/HOLD/SELL).
    """

    def __init__(
        self,
        settings: Settings,
        state_size: int,
        action_size: int = 3,
    ) -> None:
        self.settings = settings
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters from settings with sensible defaults.
        model_cfg = settings.model
        self.learning_rate: float = getattr(model_cfg, "ddqn_learning_rate", 1e-3)
        self.gamma: float = getattr(model_cfg, "ddqn_gamma", 0.99)
        self.epsilon: float = getattr(model_cfg, "ddqn_epsilon", 1.0)
        self.epsilon_min: float = getattr(model_cfg, "ddqn_epsilon_min", 0.01)
        self.epsilon_decay: float = 0.995  # multiplicative decay per episode
        self.batch_size: int = getattr(model_cfg, "batch_size", 64)
        self.target_update_freq: int = 10  # episodes between target syncs
        self.buffer_size: int = 100_000

        # Networks.
        self.policy_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.update_target_network()  # start in sync

        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.learning_rate
        )

        # Replay buffer.
        self.replay_buffer: Deque[Experience] = deque(maxlen=self.buffer_size)

        logger.info(
            "DDQNAgent initialised: state_size=%d, action_size=%d, "
            "lr=%.2e, gamma=%.3f, eps=%.2f->%.2f, batch=%d, device=%s",
            state_size,
            action_size,
            self.learning_rate,
            self.gamma,
            self.epsilon,
            self.epsilon_min,
            self.batch_size,
            self.device,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action using epsilon-greedy (training) or greedy (eval).

        Args:
            state: 1-D numpy observation vector.
            training: If ``True``, apply epsilon-greedy exploration.

        Returns:
            Integer action index.
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.policy_network.eval()
        with torch.no_grad():
            q_values = self.policy_network(state_t)
        self.policy_network.train()
        return int(q_values.argmax(dim=1).item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self) -> Optional[float]:
        """Sample a mini-batch and perform one DDQN update.

        Returns:
            The mean batch loss, or ``None`` if the buffer has fewer samples
            than ``batch_size``.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q-values for the taken actions.
        current_q = self.policy_network(states_t).gather(1, actions_t).squeeze(1)

        # DDQN: action selection from policy net, evaluation from target net.
        with torch.no_grad():
            best_actions = self.policy_network(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states_t).gather(1, best_actions).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self) -> None:
        """Hard-copy policy network weights to the target network."""
        self.target_network.load_state_dict(self.policy_network.state_dict())

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(
        self,
        env: Any,  # gymnasium.Env -- typed loosely to avoid import dependency
        episodes: int = 1000,
        max_steps: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Run the full training loop over the given environment.

        Args:
            env: A Gymnasium-compatible environment (e.g.
                ``CryptoTradingEnv``).
            episodes: Number of training episodes.
            max_steps: Optional cap on steps per episode. ``None`` means run
                until the environment terminates.

        Returns:
            Dictionary with ``'episode_rewards'``, ``'episode_lengths'``,
            ``'epsilon_values'``, ``'losses'``.
        """
        history: Dict[str, List[float]] = {
            "episode_rewards": [],
            "episode_lengths": [],
            "epsilon_values": [],
            "losses": [],
        }

        best_reward = -float("inf")

        logger.info("Starting DDQN training for %d episodes.", episodes)

        for episode in range(1, episodes + 1):
            obs, _info = env.reset()
            total_reward = 0.0
            step_count = 0
            episode_losses: List[float] = []

            done = False
            while not done:
                action = self.act(obs, training=True)
                next_obs, reward, terminated, truncated, _info = env.step(action)
                done = terminated or truncated

                self.remember(obs, action, reward, next_obs, done)
                loss = self.replay()
                if loss is not None:
                    episode_losses.append(loss)

                obs = next_obs
                total_reward += reward
                step_count += 1

                if max_steps is not None and step_count >= max_steps:
                    break

            # Decay epsilon.
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Periodically sync target network.
            if episode % self.target_update_freq == 0:
                self.update_target_network()

            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            history["episode_rewards"].append(total_reward)
            history["episode_lengths"].append(float(step_count))
            history["epsilon_values"].append(self.epsilon)
            history["losses"].append(avg_loss)

            if total_reward > best_reward:
                best_reward = total_reward

            if episode % max(1, episodes // 20) == 0 or episode == 1:
                logger.info(
                    "Episode %4d/%d  reward=%+8.2f  best=%+8.2f  "
                    "eps=%.4f  loss=%.4f  steps=%d",
                    episode,
                    episodes,
                    total_reward,
                    best_reward,
                    self.epsilon,
                    avg_loss,
                    step_count,
                )

        logger.info(
            "DDQN training complete.  Best episode reward: %+.2f", best_reward
        )
        return history

    # ------------------------------------------------------------------
    # Greedy inference
    # ------------------------------------------------------------------
    def predict(self, state: np.ndarray) -> int:
        """Select the greedy action (no exploration).

        Args:
            state: 1-D numpy observation vector.

        Returns:
            Integer action index.
        """
        return self.act(state, training=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_model(self, path: str | Path) -> None:
        """Save agent state (networks, optimiser, hyperparams) to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "policy_state_dict": self.policy_network.state_dict(),
            "target_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
        }
        torch.save(checkpoint, path)
        logger.info("DDQNAgent saved to %s", path)

    def load_model(self, path: str | Path) -> None:
        """Load a previously saved agent checkpoint from *path*."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.state_size = checkpoint["state_size"]
        self.action_size = checkpoint["action_size"]
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.gamma = checkpoint.get("gamma", self.gamma)
        self.learning_rate = checkpoint.get("learning_rate", self.learning_rate)

        self.policy_network = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = QNetwork(self.state_size, self.action_size).to(self.device)

        self.policy_network.load_state_dict(checkpoint["policy_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_state_dict"])

        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.learning_rate
        )
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.policy_network.eval()
        self.target_network.eval()
        logger.info("DDQNAgent loaded from %s", path)
