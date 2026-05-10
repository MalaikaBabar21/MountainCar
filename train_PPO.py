import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import torch
import random
import os

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def make_env():
    env = gym.make("MountainCarContinuous-v0")
    env = Monitor(env, filename="./logs/monitor.csv")
    env.reset(seed=SEED)
    return env

train_env = DummyVecEnv([make_env])

train_env = VecNormalize(
    train_env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
)
eval_env = DummyVecEnv([make_env])

eval_env = VecNormalize(
    eval_env,
    norm_obs=True,
    norm_reward=False,
    clip_obs=10.0,
)

eval_env.obs_rms = train_env.obs_rms

model = PPO(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,


    policy_kwargs=dict(
        net_arch=[256, 256]
    ),


    verbose=1,
    seed=SEED,
    device="auto",
)


eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False,
)

TIMESTEPS = 300_000

model.learn(
    total_timesteps=TIMESTEPS,
    callback=eval_callback,
    progress_bar=True,
)

model.save("ppo_mountaincar_continuous")
train_env.save("vec_normalize.pkl")

print("\nTraining complete.")
print("Best model saved in ./best_model/")

