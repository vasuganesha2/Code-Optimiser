from env.env import CompilerEnv, Action
from env.tasks import get_tasks

# Reordered to let CSE/Propagations happen before the Folder masks them with constants
PASS_SEQUENCE = [
    "copy_prop",
    "local_cse",
    "global_cse",
    "store_load_fwd",
    "dead_store_elim",
    "const_fold",
    "dead_code_elim",
    "lcm",
    "code_motion",
]

def run_task(task):
    env = CompilerEnv(task)
    obs = env.reset()

    total_reward = 0.0
    initial_count = obs.num_instructions

    pass_index = 0
    no_progress_streak = 0

    # The loop should run until convergence or max_steps
    for _ in range(env.max_steps):
        # If we have tried every single pass in the sequence and none were 'useful', stop.
        if no_progress_streak >= len(PASS_SEQUENCE):
            obs, reward, done, info = env.step(Action(pass_name="stop"))
            total_reward += reward
            break

        action_name = PASS_SEQUENCE[pass_index % len(PASS_SEQUENCE)]
        obs, reward, done, info = env.step(Action(pass_name=action_name))
        total_reward += reward

        # Use the 'useful' flag from your env.step info.
        # A pass is useful if it changed the IR, even if the instruction count stayed the same.
        if info.get("useful"):
            no_progress_streak = 0
            # We don't reset pass_index to 0 here so that we 
            # continue through the pipeline (reaching global/lcm passes).
        else:
            no_progress_streak += 1

        pass_index += 1

        if done:
            break

    final_state = env.state()
    score = task["grader"](obs)

    return {
        **final_state,
        "start_instructions": initial_count,
        "total_reward": round(total_reward, 4),
        "grade": round(score, 4),
    }

def run_baseline():
    print(f"{'Task':<16} {'Start':>6} {'End':>6} {'Removed':>8} {'Grade':>7}  History")
    print("-" * 90)

    for task in get_tasks():
        result = run_task(task)
        start = result["start_instructions"]
        end = result["num_instructions"]

        # Formatting history for better readability
        history_str = ", ".join(result["history"]) if result["history"] else "None"

        print(
            f"{task['name']:<16} {start:>6} {end:>6} {start - end:>8} "
            f"{result['grade']:>7.2f}  [{history_str}]"
        )

if __name__ == "__main__":
    run_baseline()