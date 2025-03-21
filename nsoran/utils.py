# -- Public Imports
import os
import math
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

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


class ActionMapperActorCritic:
    def __init__(self, minVal, maxVal):
        self.minVal = minVal
        self.maxVal = maxVal

    def map(self, action):
        return np.clip(np.round(action), self.minVal, self.maxVal).astype(int)

    def round_and_clip_action(self, action):
        if isinstance(action, tf.Tensor):
            action = action.numpy().flatten().tolist()  # Convert Tensor to list
        elif isinstance(action, np.ndarray):
            action = action.flatten().tolist()  # Convert np.ndarray to list
        elif isinstance(action, list):
            action = [float(a) for a in action]  # Ensure all elements are floats

        return [max(self.minVal, min(self.maxVal, round(val))) for val in action]

    def minmax_scale_action(self, action):
        """
        Manually scale the action values to be integers between minVal and maxVal.
        Handles edge cases where all action values are the same.
        """
        # Ensure the input is in list format and convert to a numpy array
        if isinstance(action, tf.Tensor):
            action = action.numpy().flatten().tolist()  # Convert Tensor to list
        elif isinstance(action, np.ndarray):
            action = action.flatten().tolist()  # Convert np.ndarray to list
        elif isinstance(action, list):
            action = [float(a) for a in action]  # Ensure all elements are floats

        # Find the min and max of the action list
        min_action = min(action)
        max_action = max(action)

        # Check if min and max are equal (no variance in input)
        if min_action == max_action:
            # In this case, return a fixed action value within the range
            return [self.minVal] * len(action)  # or [self.maxVal] or any fixed value in range

        # Scale the action values between 0 and 1
        normalized_action = [(a - min_action) / (max_action - min_action) for a in action]

        # Scale the normalized action to the desired range [minVal, maxVal]
        scaled_action = [
            int(np.round(self.minVal + (self.maxVal - self.minVal) * norm_a)) for norm_a in normalized_action
        ]

        # Clip the values to ensure they stay within the range [minVal, maxVal]
        clipped_action = [max(self.minVal, min(self.maxVal, val)) for val in scaled_action]

        return clipped_action


# Ornstein-Uhlenbeck process for generating noise
class OUActionNoise:
    def __init__(self, mean=1, std_deviation=float(0.2)*np.ones(1), theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


def update_activate_cost(curr_tds, actions):

    curr_tds = [td + 100 if act>10 else 0 for td, act in zip(curr_tds, actions)]
    activate_costs = [0.9 ** (0.01 * td) for td in curr_tds]
    return curr_tds, activate_costs


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
