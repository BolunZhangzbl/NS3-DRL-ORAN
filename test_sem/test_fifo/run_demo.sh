#!/bin/bash

# Compile C++ code
g++ cpp_proc.cc -o cpp_proc

# Start C++ first (foreground), then Python (background)
./cpp_proc &
python3 python_proc.py &

# Wait for background processes
wait

# Cleanup
rm -f cpp_proc
echo "Collaboration complete!"
