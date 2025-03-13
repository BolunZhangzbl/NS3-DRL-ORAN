# -- Public Imports
import argparse
import tensorflow as tf

# -- Private Imports
from nsoran.runner.dqn_runner import DQNRunner
from nsoran.config import get_config

# -- Global Variables


# -- Functions

def main():
    args = get_config()

    # cuda
    if args.use_cuda and tf.config.list_physical_devices('GPU'):
        print("Using GPU for computation...")
        device = tf.device("GPU:0")
    else:
        print("Using CPU for computation...")
        device = tf.device("CPU")

    dqn_runner = DQNRunner(args)
    dqn_runner.run()


if __name__ == '__main__':
    main()
