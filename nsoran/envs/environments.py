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

    def __init__(self, args, world=None, reset_callback=None, reward_callback=None, observation_callback=None,
                 info_callback=None, done_callback=None, post_step_callback=None):

        self.num_enbs = args.num_enbs
        self.world = world
        self.current_step = 0
        self.time = 0
        self.done = False

        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.post_step_callback = post_step_callback

        # argsure spaces
        self.action_space = []
        self.observation_space = []

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

        next_state = self._get_obs()
        reward = self._get_reward()
        self.done = self._get_done(reward)

        return next_state, reward, self.done

    def reset(self):
        self.current_step = 0
        self.done = False

    def _get_obs(self):
        df_state = self.data_parser.read_kpms('state')

        # Add Tx power from ORAN scenario
        max_bytes = self.args.num_enbs*3 - 1
        data_tx_power = os.read(self.fifo1, max_bytes)
        try:
            data_tx_power = data_tx_power.decode("uef-8")
        except UnicodeDecodeError:
            data_tx_power = ("60,"*self.args.num_enbs)[:-1]

        # Map str to list int
        data_tx_power = list(map(int, data_tx_power))
        df_state['tx_power'] = data_tx_power

        data_state = df_state.drop(columns=['time', 'cellId', 'IMSI', 'ccId']).to_numpy()

        return data_state

    def _get_reward(self):
        df_reward = self.data_parser.read_kpms('reward')

        if df_reward.empty:
            return 0

        data_reward = df_reward.drop(columns=['time', 'cellId', 'IMSI', 'ccId']).to_numpy()
        reward = np.dot(data_reward, self.reward_weights).sum()

        return reward

    def _get_done(self, reward):

        return True if reward >= self.reward_threshold else False

    def _send_action(self, action):
        enbs_active_status = np.array(action)
        txp = ','.join(enbs_active_status.astype(str))
        print(f"action taken: {txp}")
        send_action(txp, self.fifo2)



