#!/bin/bash

# Ensure all previous semaphores are deleted before running
if [ -e /dev/shm/sem.drl_ready ]; then
    rm -f /dev/shm/sem.drl_ready
fi

if [ -e /dev/shm/sem.ns3_ready ]; then
    rm -f /dev/shm/sem.ns3_ready
fi

# Start C++ first (foreground), then Python (background)
python3 run_dqn.py

# Wait for background processes

rm -f actions.json
echo "Collaboration complete!"
