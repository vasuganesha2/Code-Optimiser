from env.env import CompilerEnv, Action
from env.tasks import get_tasks

# Use only passes that are registered in env.env.AVAILABLE_PASSES
PASS_SEQUENCE = [
    "const_fold",
    "copy_prop",
    "local_cse",
    "dead_code_elim",
    "code_motion",
    "lcm",
]


def run_task(task):
    env = CompilerEnv(task)
    obs = env.reset()

    total_reward = 0.0
    initial_count = obs.num_instructions

    pass_index = 0
    no_progress_streak = 0
    prev_count = obs.num_instructions

    for _ in range(env.max_steps):
        if no_progress_streak >= len(PASS_SEQUENCE):
            obs, reward, done, info = env.step(Action(pass_name="stop"))
            total_reward += reward
            break

        action_name = PASS_SEQUENCE[pass_index % len(PASS_SEQUENCE)]
        obs, reward, done, info = env.step(Action(pass_name=action_name))
        total_reward += reward

        new_count = obs.num_instructions
        changed = new_count < prev_count

        if changed:
            no_progress_streak = 0
            pass_index = 0
        else:
            no_progress_streak += 1
            pass_index += 1

        prev_count = new_count

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
    print(f"{'Task':<14} {'Start':>6} {'End':>6} {'Removed':>8} {'Grade':>7}  History")
    print("-" * 80)

    for task in get_tasks():
        result = run_task(task)
        start = result["start_instructions"]
        end = result["num_instructions"]

        print(
            f"{task['name']:<14} {start:>6} {end:>6} {start - end:>8} "
            f"{result['grade']:>7.2f}  {result['history']}"
        )


if __name__ == "__main__":
    run_baseline()