from env.env import CompilerEnv, Action
from env.tasks import get_tasks
from env.graders import grade

PASS_SEQUENCE = ["const_fold", "dead_code_elim"]


def run_task(task):
    env = CompilerEnv(task)
    obs = env.reset()
    initial_count = obs.num_instructions
    total_reward  = 0.0

    pass_index         = 0
    no_progress_streak = 0

    for _ in range(env.max_steps):
        # Full cycle did nothing — stop early
        if no_progress_streak >= len(PASS_SEQUENCE):
            action = Action(pass_name="stop")
            obs, reward, done, info = env.step(action)
            total_reward += reward
            break

        action_name = PASS_SEQUENCE[pass_index % len(PASS_SEQUENCE)]
        obs, reward, done, info = env.step(Action(pass_name=action_name))
        total_reward += reward

        if info["useful"]:
            no_progress_streak = 0
            pass_index = 0          # restart cycle from const_fold
        else:
            no_progress_streak += 1
            pass_index += 1

        if done:
            break

    final_state = env.state()
    final_count = final_state["num_instructions"]
    score = grade(initial_count, final_count)

    return {
        **final_state,
        "total_reward": round(total_reward, 4),
        "grade":        round(score, 4),
    }


def run_baseline():
    print(f"{'Task':<10} {'Start':>6} {'End':>6} {'Removed':>8} {'Grade':>7}  History")
    print("-" * 65)
    for task in get_tasks():
        result = run_task(task)
        start  = len(task["program"]["instructions"])
        end    = result["num_instructions"]
        print(
            f"{task['name']:<10} {start:>6} {end:>6} {start - end:>8} "
            f"{result['grade']:>7.2f}  {result['history']}"
        )


if __name__ == "__main__":
    run_baseline()
