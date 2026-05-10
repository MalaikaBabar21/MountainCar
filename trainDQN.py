import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model_DQN import DQN
from replay_buffer_DQN import ReplayBuffer
from config_DQN import *
from env_wrapper_DQN import MountainCarRewardWrapper
from plot_training_DQN import plot_rewards
rewards_list = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("MountainCar-v0")
env = MountainCarRewardWrapper(env)

policy = DQN(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)
target = DQN(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)
target.load_state_dict(policy.state_dict())
target.eval()

optimizer = optim.Adam(policy.parameters(), lr=LR)
loss_fn = nn.SmoothL1Loss()

memory = ReplayBuffer(MEMORY_SIZE)

epsilon = EPS_START
steps = 0

best_reward = -999

def select_action(state):
    global epsilon
    if np.random.random() < epsilon:
        return np.random.randint(ACTION_SIZE)
    state = torch.FloatTensor(state).to(device)
    with torch.no_grad():
        return policy(state).argmax().item()

def train_step():

    batch = memory.sample(BATCH_SIZE)

    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).to(device)

    q_values = policy(states).gather(1, actions).squeeze()

    with torch.no_grad():
        next_q = target(next_states).max(1)[0]
        target_q = rewards + GAMMA * next_q * (1 - dones)

    loss = loss_fn(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


for episode in range(EPISODES):

    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:

        action = select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        memory.push((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        steps += 1

        if len(memory) > BATCH_SIZE:
            train_step()

        if steps % TARGET_SYNC == 0:
            target.load_state_dict(policy.state_dict())

    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    print(f"Episode {episode} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

    rewards_list.append(total_reward)

    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(policy.state_dict(), "best_model.pth")
        print("Best Model Updated")

env.close()
plot_rewards(rewards_list)