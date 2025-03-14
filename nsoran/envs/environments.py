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

        self.num_enbs = args.num_enbs
        self.current_step = 0
        self.time = 0
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
        self.reward_weights = [0.4, 0.4, 0.1, 0.1]
        self.reward_threshold = int(1e6)

        # link to fifo
        self.fifo1 = os.open("/home/bolun/ns-3-dev/fifo1", os.O_RDONLY)
        self.fifo2 = os.open("/home/bolun/ns-3-dev/fifo2", os.O_WRONLY)
        print("Opening FIFOs to send/recive...")

    def step(self, action):

        self.current_step += 1
        assert len(action) == self.num_enbs

        next_state, lastest_time = self._get_obs()
        reward, _ = self._get_reward()
        self.done = self._get_done(reward)

        self._send_action(action, lastest_time)

        return next_state, reward, self.done

    def reset(self):
        self.current_step = 0
        self.done = False

    def _get_obs(self):
        df_state, latest_time = self.data_parser.read_kpms('tp')

        # Add Tx power from ORAN scenario
        max_bytes = self.args.num_enbs * 4
        data_tx_power = os.read(self.fifo1, max_bytes)
        try:
            data_tx_power = os.read(self.fifo1, max_bytes).decode("utf-8", errors="ignore").strip()
            data_tx_power = list(map(int, data_tx_power.split(",")))
        except (OSError, UnicodeDecodeError, ValueError):
            print("Warning: Error reading Tx power, using default values.")
            data_tx_power = [60] * self.num_enbs  # Default fallback

        df_state['tx_power'] = data_tx_power

        data_state = df_state.drop(columns=['time', 'cellId', 'IMSI']).to_numpy()

        return data_state, latest_time

    def _get_reward(self):
        df_reward, latest_time = self.data_parser.read_kpms('tp')

        if df_reward.empty:
            return 0, latest_time

        data_reward = df_reward.drop(columns=['time', 'cellId', 'IMSI']).to_numpy()
        reward = np.dot(data_reward, self.reward_weights).sum()

        return reward, latest_time

    def _get_done(self, reward):

        return True if reward >= self.reward_threshold else False

    def _send_action(self, action, timestamp):
        enbs_active_status = np.array(action)
        txp = ','.join(enbs_active_status.astype(str))
        print(f"action taken: {txp}")
        send_action(txp, self.fifo2, timestamp)



