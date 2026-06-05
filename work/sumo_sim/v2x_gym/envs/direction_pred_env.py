"""
DirectionPredictionEnv — Gymnasium-compliant V2X Traffic Direction Prediction
=============================================================================
Replaces the legacy DirectionPredEnv with a proper gymnasium.Env subclass.

Observation : Flat float32 vector of shape (seq_len × n_rsu × (state_dim + n_dir),)
              Default: (60 × 4 × 12) = (2880,)
Action      : Discrete(4) — predicted dominant traffic direction (0=N, 1=E, 2=S, 3=W)
Reward      : +reward_correct × class_weight if prediction matches ground truth,
              reward_wrong otherwise.
Termination : Episode ends when the data pointer exceeds valid training indices.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DirectionPredictionEnv(gym.Env):
    """
    Gymnasium environment for V2X traffic direction trend prediction.

    The agent receives a sliding window of RSU-level state features
    (density, speed, queue, delay, …) concatenated with directional
    density counts, and must predict which compass direction (N/E/S/W)
    will see the largest traffic increase over the next trend_horizon
    timesteps.

    Parameters
    ----------
    X : np.ndarray, shape (T, n_rsu, state_dim)
        Normalised RSU state features for every timestep.
    D : np.ndarray, shape (T, n_rsu, n_dir)
        Normalised directional density counts.
    Z : np.ndarray, shape (T,), dtype int64
        Ground-truth dominant direction labels.
    valid_indices : np.ndarray
        Array of valid starting timesteps (must be ≥ seq_len).
    cfg : dict
        Configuration dictionary (same keys as CFG in top.py).
    class_weights : np.ndarray or None
        Per-class reward scaling to handle label imbalance.
    render_mode : str or None
        "human" prints to stdout, "ansi" returns a string.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 30}

    def __init__(self, X, D, Z, valid_indices, cfg,
                 class_weights=None, render_mode=None):
        super().__init__()

        self.X = X
        self.D = D
        self.Z = Z
        self.valid_indices = np.asarray(valid_indices)
        self.cfg = cfg

        self.n_rsu     = cfg["n_rsu"]
        self.n_dir     = cfg["n_dir"]
        self.state_dim = cfg["state_dim"]
        self.seq_len   = cfg["seq_len"]
        self.T         = X.shape[0]
        self.H         = cfg["trend_horizon"]
        self.min_ep    = cfg["min_ep_len"]

        self.obs_dim = self.seq_len * self.n_rsu * (self.state_dim + self.n_dir)

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(cfg["action_dim"])

        self.class_weights = (
            np.asarray(class_weights, dtype=np.float32)
            if class_weights is not None
            else np.ones(self.n_dir, dtype=np.float32)
        )
        self.reward_correct = cfg.get("reward_correct", 2.0)
        self.reward_wrong   = cfg.get("reward_wrong", -1.0)

        self.ptr = 0
        self.t   = 0
        self.render_mode = render_mode

        self._dir_names  = ["North", "East", "South", "West"]
        self._dir_arrows = ["↑", "→", "↓", "←"]

    def _get_obs(self):
        window_X = self.X[self.t - self.seq_len : self.t]
        window_D = self.D[self.t - self.seq_len : self.t]
        return np.concatenate([window_X, window_D], axis=-1).flatten().astype(np.float32)

    def _get_info(self):
        true_dir = int(self.Z[self.t]) if self.t < len(self.Z) else -1
        return {
            "true_dir": true_dir,
            "sim_time": int(self.t),
            "ptr":      int(self.ptr),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        max_start = len(self.valid_indices) - self.min_ep - 1
        if max_start < 1:
            max_start = 1
        self.ptr = int(self.np_random.integers(0, max_start))
        self.t   = int(self.valid_indices[self.ptr])

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = int(action)
        true_dir = int(self.Z[self.t])

        if action == true_dir:
            reward = self.reward_correct * float(self.class_weights[true_dir])
        else:
            reward = self.reward_wrong

        self.t   += 1
        self.ptr += 1

        terminated = (
            self.ptr >= len(self.valid_indices) - 2
            or self.t >= self.T - self.H - 2
        )
        truncated = False

        if terminated:
            obs  = np.zeros(self.obs_dim, dtype=np.float32)
            info = {"true_dir": true_dir, "sim_time": int(self.t), "ptr": int(self.ptr)}
        else:
            obs  = self._get_obs()
            info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.t >= len(self.Z):
            msg = f"t={self.t} | [episode ended]"
        else:
            true = int(self.Z[self.t])
            arrow = self._dir_arrows[true]
            name  = self._dir_names[true]
            msg = f"t={self.t} ptr={self.ptr} | true_dir={arrow} {name}"

        if self.render_mode == "human":
            print(msg)
        elif self.render_mode == "ansi":
            return msg
