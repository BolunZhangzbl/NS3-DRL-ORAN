import os
import posix_ipc
import time

# Create FIFOs if they don't exist
if not os.path.exists("./fifo1"):
    os.mkfifo("./fifo1")
if not os.path.exists("./fifo2"):
    os.mkfifo("./fifo2")

try:
    sem_cpp = posix_ipc.Semaphore("/sem_cpp_fifo")
    sem_py = posix_ipc.Semaphore("/sem_py_fifo")
except posix_ipc.ExistentialError:
    sem_cpp = posix_ipc.Semaphore("/sem_cpp_fifo", flags=posix_ipc.O_CREAT, initial_value=1)
    sem_py = posix_ipc.Semaphore("/sem_py_fifo", flags=posix_ipc.O_CREAT, initial_value=0)

print("[Python] Waiting for C++...")
for _ in range(3):
    sem_py.acquire()  # Wait for C++ to write to fifo1

    # Read from fifo1
    with open("./fifo1", "r") as f1:
        data = f1.read()
    print(f"[Python] Received: {data}")

    # Write to fifo2
    with open("./fifo2", "w") as f2:
        f2.write("[Python] Hi C++!")

    sem_cpp.release()  # Signal C++ to read fifo2
    sem_py.acquire()   # Wait for C++ to finish
    sem_cpp.release()

sem_cpp.close()
sem_py.close()
