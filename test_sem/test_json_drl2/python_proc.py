import json
import posix_ipc
import time
import os


class Agent:
    def __init__(self):
        self.action = {"message": "[Python] Action sent to C++!"}

    def act(self):
        return self.action


class Env:
    def __init__(self):
        print("\n[Python] Initializing environment...")

        # Ensure JSON files exist (create empty files if they don't)
        filename = "data.json"
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                json.dump({}, f)

        try:
            self.sem_cpp = posix_ipc.Semaphore("/sem_cpp_json_drl")
            self.sem_py = posix_ipc.Semaphore("/sem_py_json_drl")
        except posix_ipc.ExistentialError:
            print("[Python] Creating semaphores...\n")
            self.sem_cpp = posix_ipc.Semaphore("/sem_cpp_json_drl", flags=posix_ipc.O_CREAT, initial_value=0)
            self.sem_py = posix_ipc.Semaphore("/sem_py_json_drl", flags=posix_ipc.O_CREAT, initial_value=0)

    def step(self, action):
        # self.sem_py.acquire()  # Wait for C++ signal

        self._send_action(action)
        self.sem_cpp.release()

        self.sem_py.acquire()
        self._get_obs()
        self._get_reward()

        print("[Python] Notifying C++ to continue...\n")
        # self.sem_cpp.release()  # Signal C++ to proceed

    def reset(self):
        print("\n[Python] Resetting environment...\n")
        self._get_obs()

    def _get_obs(self):
        # self.sem_py.acquire()  # Wait for C++ signal
        with open("data1.json", "r") as f1:
            data1 = json.load(f1)
        print(f"[Python] Received from C++ -> State: {data1['message']}\n")

    def _get_reward(self):
        print("[Python] Processing reward...\n")

    def _send_action(self, data2):
        with open("data2.json", "w") as f2:
            json.dump(data2, f2, indent=4)
        print(f"[Python] Sent to C++ -> Action: {data2['message']}\n")

        print("[Python] Signaling C++ to read action...\n")
        # self.sem_cpp.release()  # Signal C++ that new action is available

    def close(self):
        print("\n[Python] Closing environment and releasing semaphores...\n")
        self.sem_cpp.close()
        self.sem_py.close()


class DRLRunner:
    def __init__(self):
        self.env = Env()
        self.agent = Agent()
        self.num_episodes = 2
        self.max_step = 5

    def run(self):
        print("\n[Python] Starting DRL-Agent-Environment interaction...\n")
        for ep in range(self.num_episodes):
            print(f"\n\n[Python] Episode {ep + 1}/{self.num_episodes} begins...\n")

            for step in range(self.max_step):
                action = self.agent.act()
                self.env.step(action)

            print(f"\n[Python] Episode {ep + 1} complete!\n\n")

        self.env.close()
        print("\n[Python] DRL-Agent-Environment interaction finished!\n")


if __name__ == '__main__':
    drl_runner = DRLRunner()
    drl_runner.run()
