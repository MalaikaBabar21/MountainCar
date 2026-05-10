import gymnasium as gym
import torch
import numpy as np

from model_DQN import DQN
from config_DQN import *
from env_wrapper_DQN import MountainCarRewardWrapper

env = gym.make("MountainCar-v0", render_mode="rgb_array")
env = MountainCarRewardWrapper(env)

model = DQN(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

episodes = 50
rewards = []
success = 0

for episode in range(episodes):

    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:

        state_tensor = torch.FloatTensor(state)
        action = model(state_tensor).argmax().item()

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward

    rewards.append(total_reward)

    # success condition (reaching goal)
    if state[0] >= 0.5:
        success += 1

    print(f"Episode {episode} | Reward: {total_reward}")

print("\n===== FINAL RESULTS =====")
print("Success Rate:", (success / episodes) * 100, "%")
print("Average Reward:", sum(rewards) / len(rewards))

np.save("test_rewards.npy", rewards)