#!/bin/bash

# Ensure all previous semaphores are deleted before running
if [ -e /dev/shm/sem.sem_cpp_json ]; then
    rm -f /dev/shm/sem.sem_cpp_json_drl
fi

if [ -e /dev/shm/sem.sem_py_json ]; then
    rm -f /dev/shm/sem.sem_py_json_drl
fi

# Compile C++ code
g++ cpp_proc.cc -o cpp_proc

# Start C++ first (foreground), then Python (background)
./cpp_proc &
python3 python_proc.py &

# Wait for background processes
wait

# Cleanup
rm -f cpp_proc data.json
echo "Collaboration complete!"
