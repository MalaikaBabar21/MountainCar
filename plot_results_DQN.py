import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("test_rewards.npy")

plt.plot(rewards)
plt.title("Test Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()