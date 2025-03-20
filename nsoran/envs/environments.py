# -- Public Imports
import os
import gym
import time
import json
import logging

import numpy as np
import posix_ipc

# -- Private Imports
from nsoran.data_parser import *
from nsoran.utils import *

# -- Global Variables
logging.basicConfig(level=logging.INFO)

# -- Functions

class ORANSimEnv(gym.Env):

    def __init__(self, args):
        self.active_power = args.active_power
        self.num_enb = args.num_enb
        self.num_state = args.num_state
        self.latest_time = 0
        self.done = False

        # Data from env
        self.data_parser = DataParser(args)

        # reward
        self.curr_tds = [0] * self.num_enb                 # Calculate activate cost
        self.reward_weights = [0.1, 0.01, 0.1, -0.5, -0.2]  # prbs, tp, sinr, tx_power, activate_cost
        self.reward_threshold = int(1e6)

        # JSON file paths for communication
        self.actions_file = "actions.json"  # File where DRL writes actions
        if not os.path.exists(self.actions_file):
            with open(self.actions_file, "w") as f:
                json.dump({}, f)

        # Semaphore connections
        try:
            self.ns3_ready = posix_ipc.Semaphore("/ns3_ready")
            self.drl_ready = posix_ipc.Semaphore("/drl_ready")
        except posix_ipc.ExistentialError:
            self.ns3_ready = posix_ipc.Semaphore("/ns3_ready", flags=posix_ipc.O_CREAT, initial_value=0)
            self.drl_ready = posix_ipc.Semaphore("/drl_ready", flags=posix_ipc.O_CREAT, initial_value=0)

        print("Initializing environment with JSON-based communication...")

    def step(self, action):
        assert len(action) == self.num_enb
        # Send action to NS-3
        self._send_action(action)
        self.drl_ready.release()  # Signal DRL that it is ready for new data

        print("Waiting for ns3_ready")
        self.ns3_ready.acquire()  # Block until NS-3 signals it's ready
        next_state = self._get_obs(action)  # Wait for NS-3 update, then get new state
        reward = self._get_reward(next_state, action)
        self.done = self._get_done(reward)

        return next_state, reward, self.done

    def reset(self):
        self.done = False
        state = self._get_obs(None)

        return state

    def close(self):
        self.ns3_ready.close()
        self.drl_ready.close()

    def _get_obs(self, action=None):
        df_state = self.data_parser.aggregate_kpms()
        self.latest_time = self.data_parser.last_read_time

        # Add Tx power from ORAN scenario
        # data_tx_power = self._read_tx_power_json()
        action = [self.active_power if val else 0 for val in action] if action is not None else [self.active_power] * self.num_enb
        df_state['tx_power'] = action

        # Add activate cost to state
        self.curr_tds, df_state['ac'] = update_activate_cost(self.curr_tds, action)

        df_state = df_state.drop(columns=['cellId'], errors='ignore')
        print("df_state:")
        print(df_state)

        data_state = df_state.to_numpy().flatten()

        return data_state

    def _get_reward(self, data_state, action):
        data_reward = data_state.reshape(4, self.num_state)
        reward = np.dot(data_reward, self.reward_weights).sum()

        # Minus the number of active enbs
        reward -= self.reward_weights[1] * sum(action)

        return reward

    def _get_done(self, reward):

        return True if reward >= self.reward_threshold else False

    # def _read_tx_power_json(self):
    #     """Read Tx power data from JSON file."""
    #     try:
    #         with open(self.tx_power_file, 'r') as f:
    #             tx_power_data = json.load(f)
    #             # Make sure to return the power values
    #             return tx_power_data if isinstance(tx_power_data, list) else [self.active_power] * self.num_enb
    #     except (FileNotFoundError, json.JSONDecodeError) as e:
    #         print(f"Warning: Error reading {self.tx_power_file}: {e}, using default values.")
    #     return [self.active_power] * self.num_enb

    def _send_action(self, action):
        """Write the action to the JSON file."""
        if isinstance(action, np.ndarray):
            action = action.tolist()

        action_data = {'actions': action}  # Converting the action to a list for JSON storage

        try:
            with open(self.actions_file, 'w') as f:
                json.dump(action_data, f)  # Write the action data to the file
                print(f"Action taken: {action_data}")
        except OSError as e:
            print(f"Error writing to {self.actions_file}: {e}")

