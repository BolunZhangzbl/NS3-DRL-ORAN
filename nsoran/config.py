# -- Public Imports
import argparse

# -- Private Imports

# -- Global Variables


# -- Functions

def get_config():
    """
    The configuration parser

    Env parameters
        --adsf

    ORAN parameters:
        --asdf

    DRL parameters:
        --asdf
    """
    parser = argparse.ArgumentParser(description="nsoran")

    # Env parameters
    parser.add_argument('--use_cude', type=bool, action='store_false', defaul=False, help="Whether to use cuda")
    parser.add_argument('--use_wandb', type=bool, action='store_false', default=False, help='Whether to use wandb')

    # ORAN parameters
    parser.add_argument('--num_enbs', type=int, default=4, help="Number of eNBs")
    parser.add_argument('--num_ues', type=int, default=3, help="Number of UEs per eNB")
    parser.add_argument('--it_interval', type=int, default=100, help="Interaction Interval between ORAN and agent in (ms)")
    parser.add_argument('--sim_time', type=int, default=30, help="Simulation time in (sec)")

    # DRL parameters
    parser.add_argument('--max_step', type=int, default=100, help='Maximum number of steps per episode')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes for training')
    parser.add_argument('--last_n', type=int, default=10, help='Number of last episodes to consider for evaluation')
    parser.add_argument('--actor_lr', type=float, default=3e-4, help='Learning rate for the actor network')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='Learning rate for the critic network')
    parser.add_argument('--dqn_lr', type=float, default=1e-3, help='Learning rate for the DQN network')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for future rewards')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate (epsilon-greedy strategy)')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Minimum exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.9999, help='Decay rate for exploration rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    return parser
