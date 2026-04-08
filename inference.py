"""
Inference script for Compiler Optimization Environment.
Follows OpenEnv stdout spec: [START], [STEP]*, [END]
"""

import os
import json
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
      Step 3: dead_code_elim → const[20]->y              (x is never used, removed)
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

def choose_pass_deterministic(obs: dict, failed: set) -> str:
    instructions = obs.get("program", [])

    # Build const map
    const_map = {}
    for inst in instructions:
        if inst["op"] == "const" and isinstance(inst["args"][0], int):
            const_map[inst["out"]] = inst["args"][0]

    # All vars used as inputs
    used_as_input = set()
    for inst in instructions:
        for arg in inst["args"]:
            if isinstance(arg, str):
                used_as_input.add(arg)

    # Live output = last instruction's output (never remove it)
    live_out = {instructions[-1]["out"]} if instructions else set()

    # Rule 1: const_fold — only if there's a non-const op with ALL resolvable int args
    can_fold = False
    for inst in instructions:
        if inst["op"] in ("const", "noop", "stop"):
            continue
        resolved = [
            const_map[a] if isinstance(a, str) and a in const_map else a
            for a in inst["args"]
        ]
        if all(isinstance(a, int) for a in resolved):
            can_fold = True
            break

    if can_fold and "const_fold" not in failed:
        return "const_fold"

    # Rule 2: dead_code_elim — only if a non-live output is never used as input
    can_dce = any(
        inst["out"] not in used_as_input and inst["out"] not in live_out
        for inst in instructions
    )

    if can_dce and "dead_code_elim" not in failed:
        return "dead_code_elim"

    # Nothing to do
    return "stop"

def call_llm_proxy() -> str:
    base_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("API_KEY")

    if not base_url or not api_key:
        print("[DEBUG] Skipping LLM proxy call locally; API vars not set")
        return ""

    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Pick any compiler pass for validation"},
        ],
        temperature=0.0,
    )
    print("[DEBUG] LLM proxy call successful")
    return resp.choices[0].message.content.strip()


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Do the proxy call once at the start
    _ = call_llm_proxy()

    # Define the tasks you created in tasks.py
    tasks_to_run = ["easy", "medium", "hard"]

    # Loop through each task sequentially
    for current_task in tasks_to_run:
        rewards:     list = []
        steps_taken: int  = 0
        score:       float = 0.0
        success:     bool  = False
        initial_count: int = 0

        # Log the START for the current task
        log_start(task=current_task, env=BENCHMARK, model=MODEL_NAME)

        try:
            resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task_name": current_task}, timeout=10)
            resp.raise_for_status()
            obs = resp.json()
            initial_count = obs["num_instructions"]

            tried_passes = set()          # track what failed this program state
            prev_num_inst = initial_count

            for step in range(1, MAX_STEPS + 1):
                action_name = choose_pass_deterministic(obs, tried_passes)

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
            print(f"[DEBUG] Fatal error on task {current_task}: {exc}", flush=True)

        finally:
            # Log the END for the current task
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()