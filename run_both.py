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

    Args:
        args: Command-line arguments for the simulation.
        stream_ns3 (bool): If True, stream NS-3 output to the console in real-time.
                             If False, suppress NS-3 output.
    """
    command = [
        "./waf", "--run",
        f"scenario_sleep --num_enb={args.num_enb} --ue_per_enb={args.ue_per_enb} "
        f"--it_period={args.it_period} --sim_time={args.sim_time}"
    ]
    print(f"Running simulation: {' '.join(command)}")
    
    stream_ns3 = args.stream_ns3
    # Run NS-3 in a separate process
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE if stream_ns3 else subprocess.DEVNULL,  # Capture stdout or suppress it
        stderr=subprocess.PIPE if stream_ns3 else subprocess.DEVNULL,  # Capture stderr or suppress it
        universal_newlines=True  # Ensure output is treated as text
    )

    if stream_ns3:
        # Stream stdout and stderr to the console in real-time
        def stream_pipe(pipe, pipe_name):
            for line in pipe:
                print(f"[NS-3 {pipe_name}] {line}", end='')

        # Create threads to stream stdout and stderr
        stdout_thread = threading.Thread(target=stream_pipe, args=(process.stdout, "stdout"))
        stderr_thread = threading.Thread(target=stream_pipe, args=(process.stderr, "stderr"))

        # Start the threads
        stdout_thread.start()
        stderr_thread.start()

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
        time.sleep(0.1)

        # run_drl(args)

        # Step 2: Wait for NS-3 to start
        # if ns3_process.poll() is not None:
        #     print("Error: NS-3 simulation exited too early!")
        #     exit(1)

        # Step 3: Start the DRL agent in a separate thread
        drl_thread = threading.Thread(target=run_drl, args=(args,))
        drl_thread.start()

        # Step 4: Monitor both NS-3 and DRL
        monitor_process(ns3_process, drl_thread)

        # Step 5: Wait for NS-3 process to complete
        stdout, stderr = ns3_process.communicate()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        print("NS-3 simulation has completed.")

    except Exception as e:
        print(f"An error occurred: {e}")
        ns3_process.kill()
        sys.exit(1)


if __name__ == '__main__':
    main()
