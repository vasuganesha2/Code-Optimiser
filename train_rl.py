# train_rl.py
import os
import requests
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7861")


class HttpCompilerEnv(gym.Env):
    """
    Gymnasium wrapper for the FastAPI compiler environment.
    Supports multi-task training (all tasks).
    """

    def __init__(self):
        super().__init__()

        # -----------------------------
        # Actions (compiler passes)
        # -----------------------------
        self.actions = [
            "const_fold",
            "dead_code_elim",
            "code_motion",
            "copy_prop",
            "local_cse",
            "global_cse",
            "lcm",
            "stop",
        ]
        self.action_space = spaces.Discrete(len(self.actions))

        # -----------------------------
        # Observation space
        # -----------------------------
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([1000, 100, 1000, 1000], dtype=np.float32),
            dtype=np.float32,
        )

        # -----------------------------
        # Load all tasks from backend
        # -----------------------------
        resp = requests.get(f"{ENV_BASE_URL}/tasks")
        resp.raise_for_status()
        self.all_tasks = resp.json()["tasks"]

        self.rng = np.random.default_rng()
        self.task_order = self.all_tasks.copy()
        self.task_idx = 0

    # -----------------------------
    # Feature Extraction
    # -----------------------------
    def _extract_features(self, obs_json):
        num_inst = obs_json.get("num_instructions", 0)
        steps_left = obs_json.get("steps_left", 0)
        instructions = obs_json.get("program", [])
        live_vars = set(obs_json.get("live_vars", []))

        used_inputs = set()
        for inst in instructions:
            for arg in inst.get("args", []):
                if isinstance(arg, str):
                    used_inputs.add(arg)

        num_foldable = 0
        num_dead = 0

        for inst in instructions:
            op = inst.get("op")

            # Foldable operations
            if op in {"add", "sub", "mul", "div", "mod", "shl", "shr", "and", "or", "xor"}:
                if all(isinstance(a, int) for a in inst.get("args", [])):
                    num_foldable += 1

            # Dead code detection
            if op not in {"store", "print", "call", "jmp", "br", "cbr", "ret", "stop", "noop"}:
                out_var = inst.get("out")
                if out_var is not None and out_var not in used_inputs and out_var not in live_vars:
                    num_dead += 1

        return np.array([num_inst, steps_left, num_foldable, num_dead], dtype=np.float32)

    # -----------------------------
    # Task Scheduler (balanced)
    # -----------------------------
    def _next_task(self):
        if self.task_idx >= len(self.task_order):
            self.rng.shuffle(self.task_order)
            self.task_idx = 0

        task = self.task_order[self.task_idx]
        self.task_idx += 1
        return task

    # -----------------------------
    # Gym API
    # -----------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options and "task_name" in options:
            task_name = options["task_name"]
        else:
            task_name = self._next_task()

        resp = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_name": task_name},
        )
        resp.raise_for_status()

        obs_json = resp.json()
        obs = self._extract_features(obs_json)

        return obs, {"task_name": task_name}

    def step(self, action_index):
        action_name = self.actions[action_index]

        resp = requests.post(
            f"{ENV_BASE_URL}/step",
            json={"pass_name": action_name},
        )
        resp.raise_for_status()

        data = resp.json()

        obs_json = data["observation"]
        reward = float(data["reward"])
        done = bool(data["done"])

        obs = self._extract_features(obs_json)

        terminated = done
        truncated = obs_json.get("steps_left", 1) <= 0

        return obs, reward, terminated, truncated, data.get("info", {})


# -----------------------------
# Training Loop
# -----------------------------
def main():
    print("Starting multi-task RL training...")

    env = HttpCompilerEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.001,
        n_steps=2048,
        batch_size=64,
    )

    model.learn(total_timesteps=50000)

    model.save("ppo_compiler_agent")
    print("Model saved!")

    # -----------------------------
    # Evaluation on all tasks
    # -----------------------------
    print("\nEvaluating on all tasks:\n")

    for task_name in env.all_tasks:
        obs, _ = env.reset(options={"task_name": task_name})

        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            total_reward += reward

        print(f"{task_name:20s} → reward = {total_reward:.2f}")


if __name__ == "__main__":
    main()