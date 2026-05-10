import imageio
import numpy as np

def save_gif(frames, filename="agent.gif"):
    imageio.mimsave(filename, frames, fps=30)