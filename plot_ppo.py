import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./logs/monitor.csv", skiprows=1)

rewards = data["r"]
timesteps = data["l"].cumsum()

plt.figure(figsize=(10,5))
plt.plot(timesteps, rewards)

plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("PPO Training Curve - MountainCarContinuous")

plt.grid()

plt.savefig("reward_plot.png", dpi=300)
plt.show()

print("Plot saved as reward_plot.png")