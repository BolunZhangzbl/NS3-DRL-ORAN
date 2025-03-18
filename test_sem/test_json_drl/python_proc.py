import json
import posix_ipc
import time
import os

class Agent:
    def __init__(self):
        self.action = {"message": "[Python] Hi C++!"}

    def act(self):
        return self.action

class Env:

    def __init__(self):
        # Ensure JSON files exist (create empty files if they don't)
        if not os.path.exists("data1.json"):
            with open("data1.json", "w") as f:
                json.dump({}, f)
        if not os.path.exists("data2.json"):
            with open("data2.json", "w") as f:
                json.dump({}, f)

        try:
            self.sem_cpp = posix_ipc.Semaphore("/sem_cpp_json_drl")
            self.sem_py = posix_ipc.Semaphore("/sem_py_json_drl")
        except posix_ipc.ExistentialError:
            self.sem_cpp = posix_ipc.Semaphore("/sem_cpp_json_drl", flags=posix_ipc.O_CREAT, initial_value=1)
            self.sem_py = posix_ipc.Semaphore("/sem_py_json_drl", flags=posix_ipc.O_CREAT, initial_value=0)

    def step(self, action, step):
        print("[Python] waiting for C++...")
        if step==0:
            self.sem_py.acquire()

        self._send_action(action)

        self._get_obs()
        self._get_reward()

        self.sem_cpp.release()

    def reset(self):
        self._get_obs()

    def _get_obs(self):
        self.sem_py.acquire()
        # Read from data1.json
        with open("data1.json", "r") as f1:
            data1 = json.load(f1)
        print(f"[Python] Received: State Info {data1['message']}")

    def _get_reward(self):
        print("[Python] We got reward!!!")

    def _send_action(self, data2):
        # Write to data2.json
        # data2 = {"message": "[Python] Hi C++!"}
        with open("data2.json", "w") as f2:
            json.dump(data2, f2, indent=4)  # Pretty-print with 4 spaces
        print(f"[Python] Sent: Action Info {data2['message']}")

        self.sem_cpp.release()

    def close(self):
        self.sem_cpp.close()
        self.sem_py.close()


class DRLRunner:
    def __init__(self):
        self.env = Env()
        self.agent = Agent()
        self.num_episodes = 3
        self.max_step = 10

    def run(self):
        for ep in range(self.num_episodes):
            self.env.reset()   # requires ns3_ready

            for step in range(self.max_step):
                action = self.agent.act()

                self.env.step(action, step)

        self.env.close()


if __name__ == '__main__':
    drl_runner = DRLRunner()
    drl_runner.run()
