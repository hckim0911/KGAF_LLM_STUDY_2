import numpy as np

def ambient_func(t):
    return 25 - 10*np.tanh(1.5*np.sin(2 * np.pi * t / 150)) 