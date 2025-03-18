#!/bin/bash

# Compile C++ code
g++ sem.cc -o sem

# Start C++ first (foreground), then Python (background)
./sem & 
python3 python_proc.py &

# Wait for all background jobs to finish
wait

# Cleanup
rm -f sem
echo "Collaboration complete!"
