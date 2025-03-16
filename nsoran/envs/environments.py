# -- Public Imports
import gym
import numpy as np


# -- Private Imports
from nsoran.data_parser import *
from nsoran.utils import *

# -- Global Variables


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

        # link to fifo
        self.fifo1 = os.open("/home/bolun/ns-3-dev/fifo1", os.O_RDONLY)
        self.fifo2 = os.open("/home/bolun/ns-3-dev/fifo2", os.O_WRONLY)
        print("Opening FIFOs to send/receive...")

    def step(self, action):
        assert len(action) == self.num_enb

        self._send_action(action)                 # Send action to ns-3

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
        data_tx_power = self._read_fifo()
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

    def _read_fifo(self):
        """Read from FIFO safely, returning default values on error."""
        try:
            if not os.path.exists(self.fifo1):
                raise FileNotFoundError(f"FIFO {self.fifo1} does not exist.")

            data = os.read(self.fifo1, 1024).decode("utf-8", errors="ignore").strip()

            if not data:
                raise ValueError("Empty FIFO data")

            power_values = [int(x) for x in data.split(",") if x.strip().isdigit()]

            return power_values if power_values else [self.active_power] * self.num_enb

        except (OSError, ValueError, FileNotFoundError) as e:
            print(f"Warning: Error reading {self.fifo1}: {e}, using default values.")

        # Return default values if there was an error
        return [self.active_power] * self.num_enb

    def _send_action(self, action):
        enbs_active_status = np.array(action)
        txp = ','.join(enbs_active_status.astype(str)) + '\n'  # Add newline for proper termination
        print(f"Action taken: {txp}")

        try:
            os.write(self.fifo2, txp.encode('utf-8'))  # Send to FIFO2
        except OSError as e:
            print(f"Error writing to FIFO2: {e}")

