# -- Public Imports
import os
import sys
import time
import subprocess

# -- Private Imports
from nsoran.runner.dqn_runner import DQNRunner
from nsoran.config import get_parser

# -- Global Variables

def run_drl(args):
    """
    Run the DRL agent to interact with NS-3 through FIFOs.
    """
    dqn_runner = DQNRunner(args)
    print("Starting DQN training...")
    dqn_runner.run()
    print("DQN training completed.")


def main():
    parser = get_parser()
    args = parser.parse_args()

    run_drl(args)


if __name__ == '__main__':
    main()
