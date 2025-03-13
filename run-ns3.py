
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
import log_gen 
import subprocess, signal
import shlex



if "waf" not in os.listdir("/home/waleed/holistic/"):
        raise Exception(f'Unable to locate ns3-gym in the folder : {"/home/waleed/holistic/"}')
        
    ## store current folder path
current_folder_path = os.getcwd()

## Copy prisma into ns-3 folder
# os.system(f'rsync -r ./home/waleed/holistic/scratch/sim2')

## go to ns3 dir
os.chdir("/home/waleed/holistic/")

## run ns3 configure
# os.system('./waf -d optimized configure')

## run NS3 simulator
run_ns3_command = shlex.split(f'./waf --run sim2')
# process =subprocess.Popen(run_ns3_command, stdout=subprocess.PIPE,
                                #  stderr=subprocess.PIPE,shell=True, cwd="/home/waleed/holistic/")
subprocess.Popen(run_ns3_command)
# data = process.communicate()

# print(data)
os.chdir(current_folder_path)