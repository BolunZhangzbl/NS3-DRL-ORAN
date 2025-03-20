# -- Public Imports
import argparse

# -- Private Imports

# -- Global Variables


# -- Functions

def get_parser():
    """
    The configuration parser

    Env parameters
        --use_cude       : Whether to use CUDA for GPU acceleration (default: False)
        --use_wandb      : Whether to log training metrics to Weights & Biases (default: False)

    ORAN parameters:
        --num_enb       : Number of eNBs (base stations) in the simulation (default: 4)
        --num_ues        : Number of UEs (user equipment) per eNB (default: 3)
        --it_interval    : Interaction Interval between ORAN and agent in milliseconds (default: 100ms)
        --sim_time       : Total simulation time in seconds (default: 30s)

    DRL parameters:
        --max_step       : Maximum number of steps per episode (default: 100)
        --num_episodes   : Total number of episodes for training (default: 1000)
        --last_n         : Number of last episodes to consider for evaluation (default: 10)
        --actor_lr       : Learning rate for the actor network in DRL (default: 3e-4)
        --critic_lr      : Learning rate for the critic network in DRL (default: 1e-3)
        --dqn_lr         : Learning rate for the DQN network (default: 1e-3)
        --gamma          : Discount factor for future rewards in reinforcement learning (default: 0.99)
        --epsilon        : Initial exploration rate for epsilon-greedy strategy (default: 1.0)
        --epsilon_min    : Minimum exploration rate during training (default: 0.01)
        --epsilon_decay  : Decay rate for exploration rate over time (default: 0.9999)
        --batch_size     : Batch size used during training (default: 256)
        --seed           : Random seed for reproducibility of results (default: 42)
    """
    parser = argparse.ArgumentParser(description="nsoran")

    # Env parameters
    parser.add_argument('--use_cuda', action='store_true', help="Enable CUDA (default: disabled)")
    parser.add_argument('--use_wandb', action='store_true', help="Enable Weights & Biases logging (default: disabled)")

    # ORAN parameters
    parser.add_argument('--num_enb', type=int, default=4, help="Number of eNBs")
    parser.add_argument('--ue_per_enb', type=int, default=3, help="Number of UEs per eNB")
    parser.add_argument('--it_period', type=int, default=100, help="Interaction Interval between ORAN and agent in (ms)")
    parser.add_argument('--sim_time', type=int, default=3, help="Simulation time in (sec)")
    parser.add_argument('--active_power', type=int, default=44, help="Power values for active status")
    parser.add_argument('--stream_ns3', action='store_true', default=True, help="Enable stream-ns3 (default: enabled)")

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
    parser.add_argument('--num_state', type=int, default=5, help='Number of states for each cell')

    return parser
