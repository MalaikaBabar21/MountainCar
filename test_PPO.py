import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

SEED = 42

def make_env():
    env = gym.make(
        "MountainCarContinuous-v0",
        render_mode="human"
    )
    env = Monitor(env)
    env.reset(seed=SEED)
    return env

test_env = DummyVecEnv([make_env])

# Load normalization stats
test_env = VecNormalize.load(
    "vec_normalize.pkl",
    test_env
)

test_env.training = False
test_env.norm_reward = False

model = PPO.load(
    "./best_model/best_model"
)
episodes = 5

for ep in range(episodes):

    obs = test_env.reset()
    done = False
    total_reward = 0

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, done, info = test_env.step(action)

        total_reward += reward[0]

    print(f"Episode {ep+1} Reward: {total_reward:.2f}")

test_env.close()