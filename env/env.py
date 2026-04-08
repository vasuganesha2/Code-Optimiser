import copy
from env.models import Observation, Action
from env.passes import const_fold, dead_code_elim, noop

# Registry of all available passes
# "stop" is handled specially in step() — it is not a real transform pass
AVAILABLE_PASSES = {
    "const_fold":     const_fold,
    "dead_code_elim": dead_code_elim,
    "noop":           noop,
    "stop":           noop,   # sentinel; logic is in step()
}


class CompilerEnv:
    def __init__(self, task):
        self.task = task
        self.max_steps = 10
        self.reset()

    def reset(self):
        # Deep copy so each episode starts fresh from the original program
        self.program = copy.deepcopy(self.task["program"])
        self.history = []
        self.step_count = 0
        self.last_action = "none"
        return self._get_obs()

    # Observation

    def _get_obs(self) -> Observation:
        return Observation(
            num_instructions=len(self.program["instructions"]),
            steps_left=self.max_steps - self.step_count,
            last_action=self.last_action,
            program=self.program["instructions"],
        )

    # Precondition checks — tells us whether a pass will actually do work

    def _can_apply(self, name: str) -> bool:
        """Return True only if the pass would change the program."""
        if name == "noop" or name == "stop":
            # noop/stop are always valid but give no reward
            return True

        if name == "const_fold":
            # Build const map first
            const_map = {}
            for inst in self.program["instructions"]:
                if inst["op"] == "const" and len(inst["args"]) == 1:
                    const_map[inst["out"]] = inst["args"][0]

            for inst in self.program["instructions"]:
                if inst["op"] in ("const", "noop", "stop"):
                    continue
                # Resolve args using const_map
                resolved = [
                    const_map[a] if isinstance(a, str) and a in const_map else a
                    for a in inst["args"]
                ]
                if all(isinstance(a, int) for a in resolved):
                    return True
            return False

        if name == "dead_code_elim":
            # Useful if at least one non-output instruction is unused
            used = self._used_vars()
            outputs = self._output_vars()
            for inst in self.program["instructions"]:
                if inst["out"] not in used and inst["out"] not in outputs:
                    return True
            return False

        return False

    def _used_vars(self) -> set:
        """Variables that appear as arguments in some instruction."""
        used = set()
        for inst in self.program["instructions"]:
            for arg in inst["args"]:
                if isinstance(arg, str):
                    used.add(arg)
        return used

    def _output_vars(self) -> set:
        """
        Variables that are program outputs (live out).
        We treat the last instruction's output as the program result
        so it is never eliminated by dead_code_elim.
        """
        if not self.program["instructions"]:
            return set()
        return {self.program["instructions"][-1]["out"]}


    def step(self, action: Action):
        if action.pass_name not in AVAILABLE_PASSES:
            raise ValueError(
                f"Unknown pass '{action.pass_name}'. "
                f"Available: {list(AVAILABLE_PASSES)}"
            )

        prev_count = len(self.program["instructions"])
        self.step_count += 1
        self.last_action = action.pass_name

        if action.pass_name == "stop":
            if not self._can_apply("const_fold") and not self._can_apply("dead_code_elim"):
                reward = +1.0   
            else:
                reward = -0.2   
            return self._get_obs(), reward, True, {"useful": False}

        if self._can_apply(action.pass_name):
            self.program = AVAILABLE_PASSES[action.pass_name](self.program)
            self.history.append(action.pass_name)
            useful = True
        else:
            useful = False

        curr_count = len(self.program["instructions"])


        instructions_removed = prev_count - curr_count
        reward = (
            instructions_removed * 1.0
            - (0.2 if not useful else 0.0)
            - 0.05
        )

        done = self.step_count >= self.max_steps

        return self._get_obs(), reward, done, {"useful": useful}

    def state(self) -> dict:
        return {
            "num_instructions": len(self.program["instructions"]),
            "history": self.history,
            "program": self.program,
        }