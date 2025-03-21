# -- Public Imports
import os
import math
import numpy as np
import tensorflow as tf

# -- Private Imports

# -- Global Variables

mcs_to_cr = {
    0:  0.094,  1:  0.122,  2:  0.154,  3:  0.192,  4:  0.242,
    5:  0.293,  6:  0.344,  7:  0.408,  8:  0.471,  9:  0.529,
    10: 0.264,  11: 0.295,  12: 0.331,  13: 0.382,  14: 0.425,
    15: 0.473,  16: 0.510,  17: 0.340,  18: 0.365,  19: 0.408,
    20: 0.436,  21: 0.487,  22: 0.521,  23: 0.567,  24: 0.612,
    25: 0.637,  26: 0.685,  27: 0.709,  28: 0.837
}

# -- Functions


def mcs_to_cr_func(mcs):
    return mcs_to_cr.get(mcs, 0.5)


def map_mcs_bits(mcs):
    if mcs<=9:
        return 2
    elif mcs<=16:
        return 4
    elif mcs<=27:
        return 6
    else:
        return 8


def update_activate_cost(curr_tds, actions):

    curr_tds = [td + 100 if act>10 else 0 for td, act in zip(curr_tds, actions)]
    activate_costs = [0.9 ** (0.01 * td) for td in curr_tds]
    return curr_tds, activate_costs


class ActionMapperActorCritic:
    def __init__(self, minVal, maxVal):
        self.minVal = minVal
        self.maxVal = maxVal

    def round_and_clip_action(self, action):
        if isinstance(action, tf.Tensor):
            action = action.numpy().flatten().tolist()  # Convert Tensor to list
        elif isinstance(action, np.ndarray):
            action = action.flatten().tolist()  # Convert np.ndarray to list
        elif isinstance(action, list):
            action = [float(a) for a in action]  # Ensure all elements are floats

        return [max(self.minVal, min(self.maxVal, round(val))) for val in action]


def save_lists(file_path, ep_rewards, step_rewards, avg_rewards, ep_losses,
               step_actor_losses, step_critic_losses):

    if os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)

    np.savetxt(os.path.join(file_path, r"ep_rewards.txt"), ep_rewards)
    np.savetxt(os.path.join(file_path, r"step_rewards.txt"), step_rewards)
    np.savetxt(os.path.join(file_path, r"avg_rewards.txt"), avg_rewards)
    np.savetxt(os.path.join(file_path, r"step_actor_losses.txt"), step_actor_losses)
    np.savetxt(os.path.join(file_path, r"step_critic_losses.txt"), step_critic_losses)

    np.savez(os.path.join(file_path, r"training_metrics.npz"),
             ep_rewards=np.array(ep_rewards),
             step_rewards=np.array(step_rewards),
             avg_rewards=np.array(avg_rewards),
             ep_losses=np.array(ep_losses),
             step_actor_losses=np.array(step_actor_losses),
             step_cricit_losses=np.array(step_critic_losses))

    print(f"Successfully saved lists in {file_path}!!!")
