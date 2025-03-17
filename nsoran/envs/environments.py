# -- Public Imports
import os
import gym
import time
import json
import logging
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
        self.latest_time = None
        self.done = False

        # Data from env
        self.data_parser = DataParser(args)

        # reward
        self.reward_weights = [0.4, 0.4, 0.1, -0.1]
        self.reward_threshold = int(1e6)

        # JSON file paths for communication
        self.tx_power_file = "/home/bolun/ns-3-dev/tx_power.json"  # File where NS-3 writes Tx power
        self.actions_file = "/home/bolun/ns-3-dev/actions.json"  # File where DRL writes actions

        # Semaphore connections
        try:
            self.ns3_ready = posix_ipc.Semaphore("/ns3_ready")
            self.drl_ready = posix_ipc.Semaphore("/drl_ready")
        except posix_ipc.ExistentialError as e:
            logging.error("Semaphore not found. Ensure NS-3 has initialized them in /dev/shm/")
            raise e

        print("Initializing environment with JSON-based communication...")

    def step(self, action):
        assert len(action) == self.num_enb

        # STEP 1: Wait for NS-3 to finish and DRL to send actions
        self.ns3_ready.acquire()                 # Block until NS-3 signals it's ready
        self._send_action(action)                 # Send action to NS-3

        # STEP 2: Wait for NS-3 to update and be ready with the next state
        self.drl_ready.release()                  # Signal DRL is ready for new data

        next_state = self._get_obs()              # Wait for ns-3 update, then get new state
        reward = self._get_reward(next_state)     # Calculate the reward based on the new state after performing action
        self.done = self._get_done(reward)        # Get done info given the reward

        return next_state, reward, self.done

    def reset(self):
        self.done = False
        state = self._get_obs()

        return state

    def _get_obs(self):
        df_state = self.data_parser.aggregate_kpms()
        self.latest_time = self.data_parser.last_read_time

        # Add Tx power from ORAN scenario
        data_tx_power = self._read_tx_power_json()
        df_state['tx_power'] = data_tx_power[:len(df_state)]
        df_state = df_state.drop(columns=['cellId'], errors='ignore')

        data_state = df_state.to_numpy().flatten()

        return data_state

    def _get_reward(self, data_state):
        data_reward = data_state.reshape(4, 4)
        reward = np.dot(data_reward, self.reward_weights).sum()

        return reward

    def _get_done(self, reward):

        return True if reward >= self.reward_threshold else False

    def _read_tx_power_json(self):
        """Read Tx power data from JSON file."""
        try:
            with open(self.tx_power_file, 'r') as f:
                tx_power_data = json.load(f)
                # Make sure to return the power values
                return tx_power_data if isinstance(tx_power_data, list) else [self.active_power] * self.num_enb
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Error reading {self.tx_power_file}: {e}, using default values.")
        return [self.active_power] * self.num_enb

    def _send_action(self, action):
        """Write the action to the JSON file."""
        action_data = {'actions': action.tolist()}  # Converting the action to a list for JSON storage

        try:
            with open(self.actions_file, 'w') as f:
                json.dump(action_data, f)  # Write the action data to the file
                print(f"Action taken: {action_data}")
        except OSError as e:
            print(f"Error writing to {self.actions_file}: {e}")

