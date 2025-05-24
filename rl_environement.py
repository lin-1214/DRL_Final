import os
import subprocess
import tempfile
import shutil
import random
from helper_functions import strip_ansi_escape_sequences, parse_abc_stats_line, detect_format

class ABCEnv:
    def __init__(self, abc_path="./abc", truth_file="ex100.truth"):
        self.abc_path = abc_path
        self.benchmark_path = truth_file
        self.benchmark_type = detect_format(truth_file)
        self.action_space = [
            "rewrite", "rewrite -z", "balance",
            "resub", "resub -z", "dc2",
            "refactor", "refactor -z"
        ]
        self._work_dir = os.path.abspath("runtime_results")
        os.makedirs(self._work_dir, exist_ok=True)
        self._current_aig = os.path.join(self._work_dir, "current.aig")
        self.reset()

    def reset(self):
        if self.benchmark_type == "truth":
            commands = [
                f"read_truth -xf {self.benchmark_path}",
                "collapse",
                "strash",
                f"write {self._current_aig}"
            ]
        elif self.benchmark_type == "aig":
            commands = [
                f"read {self.benchmark_path}",
                f"write {self._current_aig}"
            ]
        else:
            raise RuntimeError("Unknown benchmark type.")

        self._run_abc(commands)
        return self._get_observation()


    def step(self, action_index):
        command = self.action_space[action_index]
        temp_aig = os.path.join(self._work_dir, "temp.aig")

        commands = [
            f"read {self._current_aig}",
            command,
            f"write {temp_aig}"
        ]
        self._run_abc(commands)
        shutil.move(temp_aig, self._current_aig)

        obs = self._get_observation()
        reward = -obs["ands"]  # minimize gate count
        done = False
        return obs, reward, done, {}

    def _get_observation(self):
        commands = [
            f"read {self._current_aig}",
            "print_stats"
        ]
        output = self._run_abc(commands)
        stats_line = [l for l in output.splitlines() if "i/o" in l][-1]
        stats_line = strip_ansi_escape_sequences(stats_line)
        return parse_abc_stats_line(stats_line)

    def _run_abc(self, commands):
        joined = "\n".join(commands) + "\nquit\n"
        result = subprocess.run(
            [self.abc_path],
            input=joined.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode()

    def close(self):
        #shutil.rmtree(self._work_dir)
        pass

    def render(self):
        obs = self._get_observation()
        print(f"ANDs: {obs['ands']} | Levels: {obs['levels']}")

if __name__ == "__main__":
    # Create an environment instance
    env = ABCEnv(
        abc_path="./abc/abc",  # compiled binary path
        truth_file="./testbenches/EPFL/adder.aig"
    )

    print("=== Resetting environment ===")
    obs = env.reset()
    print("Initial observation:", obs)

    print("\n=== Taking random actions ===")
    for step in range(5):
        action_index = random.randint(0, len(env.action_space) - 1)
        print(f"\nStep {step+1}: Action = {env.action_space[action_index]}")
        obs, reward, done, _ = env.step(action_index)
        print("Observation:", obs)
        print("Reward:", reward)

    env.close()
