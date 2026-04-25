import os
import requests
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TASK_NAME = os.getenv("COMPILER_TASK", "easy_fold") # Try training on mixed!

class HttpCompilerEnv(gym.Env):
    """
    A Gymnasium wrapper that talks to the OpenEnv FastAPI backend.
    """
    def __init__(self, task_name="easy_fold"):
        super(HttpCompilerEnv, self).__init__()
        self.task_name = task_name
        
        # Action Space: Upgraded to use all 8 advanced optimization passes
        self.actions = [
            "const_fold", 
            "dead_code_elim", 
            "code_motion", 
            "copy_prop", 
            "local_cse", 
            "global_cse", 
            "lcm", 
            "stop"
        ]
        self.action_space = spaces.Discrete(len(self.actions))
        
        # Observation Space: [num_instructions, steps_left, num_foldable, num_dead]
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
        
        # Leverage the backend's new Data Flow Analysis directly!
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
            
            # 1. Foldable: Pure math operations with all integer arguments
            if op in {"add", "sub", "mul", "div", "mod", "shl", "shr", "and", "or", "xor"}:
                if all(isinstance(a, int) for a in inst.get("args", [])):
                    num_foldable += 1
            
            # 2. Dead Code: Protect memory (store) and control flow (jmp, br)
            if op not in {"store", "print", "call", "jmp", "br", "cbr", "ret", "stop", "noop"}:
                out_var = inst.get("out")
                # A variable is dead if it is never an input AND not protected by global liveness
                if out_var is not None and out_var not in used_inputs and out_var not in live_vars:
                    num_dead += 1

        return np.array([num_inst, steps_left, num_foldable, num_dead], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task_name": self.task_name})
        resp.raise_for_status()
        
        obs_json = resp.json()
        obs = self._extract_features(obs_json)
        return obs, {} 

    def step(self, action_index):
        action_name = self.actions[action_index]
        resp = requests.post(f"{ENV_BASE_URL}/step", json={"pass_name": action_name})
        resp.raise_for_status()
        
        data = resp.json()
        obs_json = data["observation"]
        reward = float(data["reward"])
        done = bool(data["done"])
        
        obs = self._extract_features(obs_json)
        
        terminated = done
        truncated = obs_json.get("steps_left", 1) <= 0
        
        return obs, reward, terminated, truncated, data.get("info", {})

def main():
    print(f"Setting up Compiler Environment for task: {TASK_NAME}")
    env = HttpCompilerEnv(task_name=TASK_NAME)
    
    # Initialize Proximal Policy Optimization (PPO)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
    
    print("Starting Training...")
    # 50,000 steps is great for the complex "mixed" task
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