# -- Public Imports
import os
import numpy as np
import matplotlib.pyplot as plt

# -- Private Imports
from utils import *

# -- Global Variables

dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

dict_ylabel = dict(
    step_losses="Loss",
    ep_losses='Episodic Loss',
    step_rewards='Reward',
    avg_rewards='Avg. Reward',
    ep_rewards='Episodic. Reward'
)

dict_markers = dict(
    step_losses='^-',
    ep_losses='o--',
    step_rewards='s-',
    avg_rewards='D--',
    ep_rewards='*-'
)

dict_colors = dict(
    step_losses='blue',
    ep_losses='green',
    step_rewards='red',
    avg_rewards='orange',
    ep_rewards='black'
)


# -- Functions


def plot(metric, agent_type="dqn", save=False):
    pass
