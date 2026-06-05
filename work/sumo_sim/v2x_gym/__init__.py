"""
V2X Gymnasium Environments
==========================
Gymnasium-compliant environments for V2X traffic direction prediction.

Registered Environments:
    - V2X-DirectionPred-v0 : Offline/synthetic data environment
"""

from gymnasium.envs.registration import register

register(
    id="V2X-DirectionPred-v0",
    entry_point="v2x_gym.envs.direction_pred_env:DirectionPredictionEnv",
    max_episode_steps=10000,
)

register(
    id="V2X-SumoLive-v0",
    entry_point="v2x_gym.envs.sumo_live_env:SumoLiveEnv",
    max_episode_steps=5000,
)
