import gymnasium as gym
import torch
import imageio

from model_DQN import DQN
from config_DQN import *
from env_wrapper_DQN import MountainCarRewardWrapper

env = gym.make("MountainCar-v0", render_mode="rgb_array")
env = MountainCarRewardWrapper(env)

model = DQN(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

frames = []

state, _ = env.reset()
done = False

while not done:

    frames.append(env.render())

    state_tensor = torch.FloatTensor(state)
    action = model(state_tensor).argmax().item()

    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

imageio.mimsave("final_agent.mp4", frames, fps=30)
env.close()

print("Video saved: final_agent.mp4")