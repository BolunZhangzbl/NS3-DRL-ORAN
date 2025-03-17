# -- Public Imports
import os
import sys
import time
import threading
import subprocess

# -- Private Imports
from nsoran.runner.dqn_runner import DQNRunner
from nsoran.config import get_parser

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

    # Read stdout and stderr in real-time
    while True:
        stdout_line = process.stdout.readline()
        stderr_line = process.stderr.readline()
        if stdout_line:
            print(stdout_line.decode(), end="")
        if stderr_line:
            print(stderr_line.decode(), end="")
        if stdout_line == b"" and stderr_line == b"" and process.poll() is not None:
            break

    return process


def run_drl(args):
    """
    Run the DRL agent to interact with NS-3 through FIFOs.
    """
    dqn_runner = DQNRunner(args)
    print("Starting DQN training...")
    dqn_runner.run()
    print("DQN training completed.")


def monitor_process(ns3_process, drl_thread):
    """
    Monitor both NS-3 and DRL processes, terminate the other process if one crashes
    """
    while drl_thread.is_alive():
        if ns3_process.poll() is not None:   # NS-3 exits unexpectedly
            print("Error: NS-3 exited unexpectedly. Terminate DRL ...")
            drl_thread.join()
            break
        time.sleep(1)

    # If DRL crashes, we should terminate NS-3
    if not drl_thread.is_alive():
        print("Error: DRL crashed! Terminate NS-3 process ...")
        ns3_process.terminate()   # Forcefully stop NS-3
        ns3_process.wait()        # Ensure it stops completely


def main():
    """
    Main function to run NS-3 simulation and DRL agent sequentially.
    """
    parser = get_parser()
    args = parser.parse_args()

    try:
        # Step 1: Start NS-3 simulation
        ns3_process = run_scenario(args)

        # Step 2: Wait for NS-3 to start
        if ns3_process.poll() is not None:
            print("Error: NS-3 simulation exited too early!")
            exit(1)

        # Step 3: Start the DRL agent in a separate thread
        drl_thread = threading.Thread(target=run_drl, args=(args,))
        drl_thread.start()

        # Step 4: Monitor both NS-3 and DRL
        monitor_process(ns3_process, drl_thread)

        # Step 5: Wait for NS-3 process to complete
        stdout, stderr = ns3_process.communicate()
        print(stdout.decode())
        print(stderr.decode())
        print("NS-3 simulation has completed.")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
