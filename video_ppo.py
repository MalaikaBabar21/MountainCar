import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

SEED = 42

env = gym.make(
    "MountainCarContinuous-v0",
    render_mode="rgb_array"
)

env = RecordVideo(
    env,
    video_folder="videos",
    name_prefix="ppo-mountaincar"
)

obs, _ = env.reset(seed=SEED)

from stable_baselines3.common.vec_env import DummyVecEnv

vec_env = DummyVecEnv([lambda: env])

vec_env = VecNormalize.load(
    "vec_normalize.pkl",
    vec_env
)

vec_env.training = False
vec_env.norm_reward = False

model = PPO.load(
    "./best_model/best_model"
)

obs = vec_env.reset()

done = False

while not done:

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, rewards, done, info = vec_env.step(action)

vec_env.close()

print("Single video saved in videos/")