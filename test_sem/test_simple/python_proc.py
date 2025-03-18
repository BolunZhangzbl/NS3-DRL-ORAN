import posix_ipc
import time

try:
    # Open semaphores created by C++
    sem_cpp = posix_ipc.Semaphore("/sem_cpp")  # C++ controls this
    sem_py = posix_ipc.Semaphore("/sem_py")    # Python waits on this
except posix_ipc.ExistentialError:
    # Fallback if C++ hasn't created them (unlikely if C++ starts first)
    sem_cpp = posix_ipc.Semaphore("/sem_cpp", flags=posix_ipc.O_CREAT, initial_value=1)
    sem_py = posix_ipc.Semaphore("/sem_py", flags=posix_ipc.O_CREAT, initial_value=0)

print("[Python] Waiting for C++ to start...")
for i in range(3):
    print(f"[Python] Waiting on sem_py (Step {i + 1})...")
    sem_py.acquire()  # Wait for C++ to signal via sem_py
    print(f"[Python] Working... (Step {i + 1})")
    time.sleep(1)
    print(f"[Python] Signaling C++ via sem_cpp (Step {i + 1})...")
    sem_cpp.release()  # Signal C++ via sem_cpp

sem_cpp.close()
sem_py.close()
# Uncomment if C++ doesn't unlink
# posix_ipc.Semaphore.unlink("/sem_cpp")
# posix_ipc.Semaphore.unlink("/sem_py")
