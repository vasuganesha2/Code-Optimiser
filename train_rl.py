import os
import requests
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TASK_NAME = os.getenv("COMPILER_TASK", "easy")

class HttpCompilerEnv(gym.Env):
    """
    A Gymnasium wrapper that talks to the OpenEnv FastAPI backend.
    """
    def __init__(self, task_name="easy"):
        super(HttpCompilerEnv, self).__init__()
        self.task_name = task_name
        
        # Action Space: 4 discrete actions
        self.actions = ["const_fold", "dead_code_elim", "noop", "stop"]
        self.action_space = spaces.Discrete(len(self.actions))
        
        # Observation Space: A 4D vector representing:
        # [num_instructions, steps_left, num_foldable_instructions, num_dead_variables]
        # We use a Box space with generous upper bounds.
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([1000, 100, 1000, 1000], dtype=np.float32),
            dtype=np.float32
        )

    def _extract_features(self, obs_json):
        """Converts the dynamic JSON AST into a fixed numerical vector for the RL neural net."""
        num_inst = obs_json.get("num_instructions", 0)
        steps_left = obs_json.get("steps_left", 0)
        
        instructions = obs_json.get("program", [])
        used_inputs = set()
        for inst in instructions:
            for arg in inst["args"]:
                if isinstance(arg, str):
                    used_inputs.add(arg)

        num_foldable = 0
        num_dead = 0
        
        # Count how many passes are currently viable
        for inst in instructions:
            if inst["op"] not in ("const", "noop", "stop"):
                if all(isinstance(a, int) for a in inst["args"]):
                    num_foldable += 1
            
            # If an output is never used as an input later, it's dead
            if inst["out"] not in used_inputs:
                num_dead += 1

        # The last instruction's output is technically the "return" value, so let's not count it as dead
        if instructions and instructions[-1]["out"] not in used_inputs:
            num_dead = max(0, num_dead - 1)

        return np.array([num_inst, steps_left, num_foldable, num_dead], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task_name": self.task_name})
        resp.raise_for_status()
        
        obs_json = resp.json()
        obs = self._extract_features(obs_json)
        return obs, {} # Gym expects (observation, info)

    def step(self, action_index):
        action_name = self.actions[action_index]
        resp = requests.post(f"{ENV_BASE_URL}/step", json={"pass_name": action_name})
        resp.raise_for_status()
        
        data = resp.json()
        obs_json = data["observation"]
        reward = float(data["reward"])
        done = bool(data["done"])
        
        obs = self._extract_features(obs_json)
        
        # In modern Gymnasium, 'done' is split into 'terminated' (goal reached/failed) 
        # and 'truncated' (time limit exceeded). 
        terminated = done
        truncated = obs_json.get("steps_left", 1) <= 0
        
        return obs, reward, terminated, truncated, data.get("info", {})

def main():
    print(f"Setting up Compiler Environment for task: {TASK_NAME}")
    env = HttpCompilerEnv(task_name=TASK_NAME)
    
    # Initialize Proximal Policy Optimization (PPO)
    # PPO is a highly stable, state-of-the-art RL algorithm.
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
    
    print("Starting Training...")
    # Train for 5,000 steps. Increase this to 50,000+ for harder tasks.
    model.learn(total_timesteps=50000)
    
    print("Training Complete! Saving model to 'ppo_compiler_agent.zip'")
    model.save("ppo_compiler_agent")

    # --- Quick Evaluation Run ---
    print("\nEvaluating the trained agent...")
    obs, info = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        action_name = env.actions[action]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        print(f"Agent chose: {action_name.ljust(15)} | Reward: {reward:.2f}")
        
    print(f"Episode finished with Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()