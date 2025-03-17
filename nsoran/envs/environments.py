# -- Public Imports
import os
import gym
import time
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

        # FIFO setup
        self.fifo1_path = "/home/bolun/ns-3-dev/fifo1"
        self.fifo2_path = "/home/bolun/ns-3-dev/fifo2"

        self.fifo1 = os.open("/home/bolun/ns-3-dev/fifo1", os.O_RDONLY | os.O_NONBLOCK)
        self.fifo2 = os.open("/home/bolun/ns-3-dev/fifo2", os.O_WRONLY)

        # Semaphore connections
        try:
            self.ns3_ready = posix_ipc.Semaphore("/ns3_ready")
            self.drl_ready = posix_ipc.Semaphore("/drl_ready")
        except posix_ipc.ExistentialError as e:
            logging.error("Semaphore not found. Ensure NS-3 has initialized them in /dev/shm/")
            raise e

        print("Opening FIFOs and Semaphores for communication...")

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

