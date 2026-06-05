"""
RunningNormalizeObservation — Gymnasium ObservationWrapper
=========================================================
Applies exponential moving-average normalization to observations,
matching the RunningNormalizer used in the SUMO inference scripts.
"""

import numpy as np
import gymnasium as gym


class RunningNormalizeObservation(gym.ObservationWrapper):
    """
    Wrapper that normalises observations using a running mean/variance
    with exponential momentum.

    This mirrors the ``RunningNormalizer`` class used in
    ``03_compare_simulations.py`` and ``04_simulate_week.py`` but
    packaged as a standard Gymnasium wrapper so it composes cleanly
    with other wrappers and vectorised environments.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    momentum : float
        Exponential moving average momentum (default 0.01).
    epsilon : float
        Small constant added to std to avoid division by zero.
    """

    def __init__(self, env, momentum=0.01, epsilon=1e-6):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.momentum = momentum
        self.epsilon  = epsilon
        self.mean  = np.zeros(obs_shape, dtype=np.float32)
        self.var   = np.ones(obs_shape, dtype=np.float32)
        self.count = 0

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        self.count += 1
        if self.count == 1:
            self.mean = obs.copy()
            self.var  = np.ones_like(obs)
        else:
            self.mean = (1 - self.momentum) * self.mean + self.momentum * obs
            self.var  = (1 - self.momentum) * self.var + self.momentum * (obs - self.mean) ** 2
        return (obs - self.mean) / (np.sqrt(self.var) + self.epsilon)
