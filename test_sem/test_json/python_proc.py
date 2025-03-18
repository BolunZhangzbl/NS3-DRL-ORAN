import json
import posix_ipc
import time
import os

# Ensure JSON files exist (create empty files if they don't)
if not os.path.exists("data1.json"):
    with open("data1.json", "w") as f:
        json.dump({}, f)
if not os.path.exists("data2.json"):
    with open("data2.json", "w") as f:
        json.dump({}, f)

try:
    sem_cpp = posix_ipc.Semaphore("/sem_cpp_json")
    sem_py = posix_ipc.Semaphore("/sem_py_json")
except posix_ipc.ExistentialError:
    sem_cpp = posix_ipc.Semaphore("/sem_cpp_json", flags=posix_ipc.O_CREAT, initial_value=1)
    sem_py = posix_ipc.Semaphore("/sem_py_json", flags=posix_ipc.O_CREAT, initial_value=0)

print("[Python] Waiting for C++...")
for _ in range(3):
    sem_py.acquire()  # Wait for C++ to write to data1.json

    # Read from data1.json
    with open("data1.json", "r") as f1:
        data1 = json.load(f1)
    print(f"[Python] Received: {data1['message']}")

    # Write to data2.json
    data2 = {"message": "[Python] Hi C++!"}
    with open("data2.json", "w") as f2:
        json.dump(data2, f2, indent=4)  # Pretty-print with 4 spaces

    sem_cpp.release()  # Signal C++ to read data2.json
    sem_py.acquire()   # Wait for C++ to finish
    sem_cpp.release()

sem_cpp.close()
sem_py.close()
#sem_cpp.unlink()
#sem_py.unlink()
