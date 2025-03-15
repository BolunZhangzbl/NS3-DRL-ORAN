# -- Public Imports
import gym
import numpy as np
from gym import spaces


# -- Private Imports
from nsoran.data_parser import *
from nsoran.utils import *

# -- Global Variables


# -- Functions

class ORANSimEnv(gym.Env):

    def __init__(self, args):
        self.args = args
        self.num_enbs = args.num_enbs
        self.current_step = 0
        self.time = 0
        self.latest_time = None
        self.done = False

        # scenario callbacks
        # self.reset_callback = reset_callback
        # self.reward_callback = reward_callback
        # self.observation_callback = observation_callback
        # self.info_callback = info_callback
        # self.done_callback = done_callback
        # self.post_step_callback = post_step_callback

        # Data from env
        self.data_parser = DataParser(args)

        # reward
        self.reward_weights = [0.4, 0.4, 0.1, -0.1]
        self.reward_threshold = int(1e6)

        # link to fifo
        self.fifo1 = os.open("/home/bolun/ns-3-dev/fifo1", os.O_RDONLY)
        self.fifo2 = os.open("/home/bolun/ns-3-dev/fifo2", os.O_WRONLY)
        print("Opening FIFOs to send/recive...")

    def step(self, action):

        self.current_step += 1
        assert len(action) == self.num_enbs

        next_state, lastest_time = self._get_obs()
        reward, _ = self._get_reward(next_state)
        self.done = self._get_done(reward)

        self._send_action(action, lastest_time)

        return next_state, reward, self.done

    def reset(self):
        self.current_step = 0
        self.done = False

    def _get_obs(self):
        df_state = self.data_parser.aggregate_kpms()
        self.latest_time = self.data_parser.last_read_time

        # Add Tx power from ORAN scenario
        data_tx_power = self._read_fifo(self.fifo1_path, default_value=self.args.active_power)
        df_state['tx_power'] = data_tx_power[:len(df_state)]

        data_state = df_state.to_numpy()

        return data_state, self.latest_time

    def _get_reward(self, df_state):

        data_reward = df_state.to_numpy()
        reward = np.dot(data_reward, self.reward_weights).sum()

        return reward

    def _get_done(self, reward):

        return True if reward >= self.reward_threshold else False

    def _read_fifo(self, fifo_path, default_value):
        """Read from FIFO safely, returning default values on error."""
        try:
            if not os.path.exists(fifo_path):
                raise FileNotFoundError(f"FIFO {fifo_path} does not exist.")

            with open(fifo_path, "r") as fifo:
                data = fifo.read().strip()
                return list(map(int, data.split(",")))
        except (OSError, ValueError, FileNotFoundError):
            print(f"Warning: Error reading {fifo_path}, using default values.")
            return [default_value] * self.num_enbs

    def _write_fifo(self, fifo_path, message):
        """Write data to FIFO safely."""
        try:
            with open(fifo_path, "w") as fifo:
                fifo.write(message + "\n")
        except OSError as e:
            print(f"Error writing to FIFO {fifo_path}: {e}")

    def _send_action(self, action, timestamp):
        enbs_active_status = np.array(action)
        txp = ','.join(enbs_active_status.astype(str))
        print(f"action taken: {txp}")
        send_action(txp, self.fifo2, timestamp)



