# -- Public Imports
import logging
import threading
import subprocess
import tensorflow as tf

# -- Private Imports
from nsoran.runner.dqn_runner import DQNRunner
from nsoran.config import get_config

# -- Global Variables


# -- Functions

def run_scenario(args):
    """
    Run the ns-3 simulation (scenario_sleep.cc) with the given parameters.
    """
    command = [
        "./waf", "--run",
        f"scenario_sleep --num_enb={args.num_enb} --ue_per_enb={args.ue_per_enb} "
        f"--it_period={args.it_period} --sim_time={args.sim_time}"
    ]
    print(f"Running simulation: {' '.join(command)}")
    subprocess.run(command, check=True)

def run_drl(args):

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Check for GPU availability
    if args.use_cuda:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info("Using GPU for computation...")
            device = tf.device("GPU:0")
        else:
            logger.warning("GPU requested but not available. Falling back to CPU.")
            device = tf.device("CPU")
    else:
        logger.info("Using CPU for computation...")
        device = tf.device("CPU")

    # Run DQN
    try:
        with device:
            logger.info("Initializing DQN Runner...")
            dqn_runner = DQNRunner(args)
            logger.info("Starting DQN training...")
            dqn_runner.run()
            logger.info("DQN training completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during DQN execution: {e}")
        raise

def main():
    """
    Main function to run the simulation and DRL agent simultaneously using multi-threading.
    """
    args = get_config()

    # Create a synchronization event
    start_event = threading.Event()

    # Create threads for simulation and DRL agent
    simulation_thread = threading.Thread(target=run_scenario, args=(args, start_event))
    drl_thread = threading.Thread(target=run_drl, args=(args, start_event))

    # Start threads
    simulation_thread.start()
    drl_thread.start()

    # Signal both threads to start simultaneously
    print("Starting both simulation and DRL agent simultaneously...")
    start_event.set()

    # Wait for threads to finish
    simulation_thread.join()
    drl_thread.join()

    print("Both simulation and DRL training have completed.")

if __name__ == '__main__':
    main()
