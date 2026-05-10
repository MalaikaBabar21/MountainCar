import numpy as np

class MountainCarRewardWrapper:
    def __init__(self, env):
        self.env = env
        self.prev_position = None

    def reset(self):
        state, info = self.env.reset()
        self.prev_position = state[0]
        return state, info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        position, velocity = state

        shaped_reward = 0

        shaped_reward += (position - self.prev_position) * 10.0
        shaped_reward += abs(velocity) * 0.1

        if position >= 0.5:
            shaped_reward += 50.0

        shaped_reward -= 0.01

        self.prev_position = position

        return state, shaped_reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()