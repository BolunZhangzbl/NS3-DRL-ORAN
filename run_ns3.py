# -- Public Imports
import os
import sys
import time
import threading
import subprocess


# -- Private Imports


# -- Global Variables


# -- Functions

def run_ns3(args):
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

    # Run NS-3 in a separate process and stream output in real-time
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,  # Capture stdout
        stderr=subprocess.PIPE,  # Capture stderr
        universal_newlines=True  # Ensure output is treated as text
    )

    # Function to stream output in real-time
    def stream_output(stream, prefix):
        for line in stream:
            print(f"{prefix}: {line}", end="")

    # Start threads to stream stdout and stderr
    stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, "NS-3 stdout"))
    stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, "NS-3 stderr"))
    stdout_thread.start()
    stderr_thread.start()

    return process
