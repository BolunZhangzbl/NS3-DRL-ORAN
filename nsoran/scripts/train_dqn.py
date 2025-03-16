# -- Public Imports
import subprocess
import time

# -- Private Imports
from nsoran.runner.dqn_runner import DQNRunner
from nsoran.config import get_config

# -- Global Variables


# -- Functions

def run_scenario(args):
    """
    Run the ns-3 simulation (scenario_sleep.cc) with the given parameters.
    This will block execution until NS-3 completes.
    """
    command = [
        "./waf", "--run",
        f"scenario_sleep --num_enb={args.num_enb} --ue_per_enb={args.ue_per_enb} "
        f"--it_period={args.it_period} --sim_time={args.sim_time}"
    ]
    print(f"Running simulation: {' '.join(command)}")

    # Run NS-3 in a separate process
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return process  # Return the process object so we can monitor it


def run_drl(args):
    """
    Run the DRL agent to interact with NS-3 through FIFOs.
    """
    dqn_runner = DQNRunner(args)
    print("Starting DQN training...")
    dqn_runner.run()
    print("DQN training completed.")


def main():
    """
    Main function to run NS-3 simulation and DRL agent sequentially.
    """
    args = get_config()

    # Step 1: Start NS-3 simulation
    ns3_process = run_scenario(args)

    # Step 2: Wait briefly to ensure NS-3 initializes
    time.sleep(1)  # Give NS-3 some time to start

    # Step 3: Start the DRL agent
    run_drl(args)

    # Step 4: Wait for NS-3 process to complete
    ns3_process.wait()
    print("NS-3 simulation has completed.")


if __name__ == '__main__':
    main()
