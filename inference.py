"""
Inference script for Compiler Optimization Environment.
Follows OpenEnv stdout spec: [START], [STEP]*, [END]
"""

import os
import textwrap
from typing import List, Optional
from openai import OpenAI
import requests

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TASK_NAME    = os.getenv("COMPILER_TASK", "easy")
BENCHMARK    = "compiler-optimization-env"
MAX_STEPS    = 10
TEMPERATURE  = 0.0   # deterministic for reproducibility

AVAILABLE_PASSES = ["const_fold", "dead_code_elim", "noop", "stop"]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert compiler optimization agent.
    You will be shown the current state of a program as a list of instructions.
    Your job is to pick the best sequence of compiler passes to minimize instruction count.

    Available passes:
    - const_fold      : replaces instructions whose ALL args are integer constants with a single const.
                        Apply MULTIPLE TIMES to propagate constants through the program.
    - dead_code_elim  : removes instructions whose output variable is never used as an input.
                        Apply AFTER const_fold to clean up now-unused variables.
    - noop            : does nothing. NEVER use this.
    - stop            : ends episode. Use ONLY when no further reduction is possible.

    STRATEGY:
    1. If any instruction has ALL integer args → use const_fold
    2. If any output variable is never referenced as an input → use dead_code_elim
    3. If neither applies → use stop

    Example:
      add [3,7]->x, mul[x,2]->y
      Step 1: const_fold  → const[10]->x, mul[x,2]->y   (x is now a known int)
      Step 2: const_fold  → const[10]->x, const[20]->y  (mul args are now all ints)
      Step 3: dead_code_elim → const[20]->y             (x is never used, removed)
      Step 4: stop

    Reply with ONLY the pass name. One of: const_fold, dead_code_elim, noop, stop
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(obs: dict, last_reward: float = 0.0) -> str:
    instructions = obs.get("program", [])
    
    # Find all variables used as inputs
    used_as_input = set()
    for inst in instructions:
        for arg in inst["args"]:
            if isinstance(arg, str):
                used_as_input.add(arg)
    
    inst_lines = "\n".join(
        f"  {i+1}. {inst['op']} {inst['args']} -> {inst['out']}"
        + (" ← DEAD (output never used)" if inst['out'] not in used_as_input else "")
        for i, inst in enumerate(instructions)
    )
    
    # Check what's possible
    all_const = [
        i+1 for i, inst in enumerate(instructions)
        if all(isinstance(a, int) for a in inst["args"]) and inst["op"] != "const"
    ]
    dead = [
        i+1 for i, inst in enumerate(instructions)
        if inst["out"] not in used_as_input
    ]
    
    hints = []
    if all_const:
        hints.append(f"→ const_fold can simplify instruction(s): {all_const}")
    if dead:
        hints.append(f"→ dead_code_elim can remove instruction(s): {dead}")
    if not all_const and not dead:
        hints.append("→ Nothing left to optimize. Use stop.")
    
    warning = ""
    if last_reward <= -0.25:
        warning = "\n⚠️ Last pass was USELESS (negative reward). Do not repeat it."
    
    return textwrap.dedent(f"""
        Current program ({obs['num_instructions']} instructions):
        {inst_lines}

        Steps remaining: {obs['steps_left']}
        Last action:     {obs['last_action']}
        Last reward:     {last_reward:.2f}{warning}

        Analysis:
        {chr(10).join(hints)}

        Which pass should be applied next?
    """).strip()


def choose_pass_llm(client: OpenAI, obs: dict, last_reward: float) -> str:
    prompt = build_user_prompt(obs, last_reward)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=10,
        )
        
        action = response.choices[0].message.content.strip().lower()
        
        for valid_pass in AVAILABLE_PASSES:
            if valid_pass in action:
                return valid_pass
                
    except Exception as e:
        print(f"[DEBUG] LLM API Error: {e}", flush=True)
        # Fallback logic if API fails so the script doesn't crash OpenEnv
        return "stop"
        
    return "stop"


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards:     list = []
    steps_taken: int  = 0
    score:       float = 0.0
    success:     bool  = False
    initial_count: int = 0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task_name": TASK_NAME}, timeout=10)
        resp.raise_for_status()
        obs = resp.json()
        initial_count = obs["num_instructions"]

        tried_passes = set()          # track what failed this program state
        prev_num_inst = initial_count
        last_reward = 0.0             # Track reward for the prompt

        for step in range(1, MAX_STEPS + 1):
            action_name = choose_pass_llm(client, obs, last_reward)

            resp = requests.post(
                f"{ENV_BASE_URL}/step",
                json={"pass_name": action_name},
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()

            obs    = result["observation"]
            reward = float(result["reward"])
            done   = bool(result["done"])

            # Update last_reward so the LLM knows if its action was useful on the next loop
            last_reward = reward

            # If pass was useless, remember it — unless program shrank (reset memory)
            if reward <= -0.1:
                tried_passes.add(action_name)
            
            # Program changed — reset tried passes so we re-evaluate
            if obs["num_instructions"] < prev_num_inst:
                tried_passes.clear()
                prev_num_inst = obs["num_instructions"]

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_name, reward=reward, done=done, error=None)

            if done:
                break

        final_count = obs["num_instructions"]
        score = (initial_count - final_count) / initial_count if initial_count > 0 else 1.0
        score = max(0.0, min(1.0, score))
        success = score > 0.0

    except Exception as exc:
        print(f"[DEBUG] Fatal error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()