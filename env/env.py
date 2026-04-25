import copy
from collections import Counter
from typing import Any, Dict, List, Set, Union

from env.models import Instruction, Observation, Action
from env.passes import (
    const_fold, 
    dead_code_elim, 
    noop, 
    loop_invariant_code_motion,
    copy_prop,
    local_cse
)
from env.ops import (
    is_math, is_terminator, is_pure, 
    is_constant, can_fold, evaluate, normalize
)



AVAILABLE_PASSES = {
    "const_fold": const_fold,
    "dead_code_elim": dead_code_elim,
    "code_motion": loop_invariant_code_motion,
    "copy_prop": copy_prop,
    "local_cse": local_cse,
    "noop": noop,
    "stop": noop,
}



class CFGNode:
    inst: Instruction
    succ: List[int]


class CompilerEnv:
    def __init__(self, task: Dict[str, Any]):
        self.task = task
        self.max_steps = 20
        self.reset()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        self.program: List[Instruction] = [
            Instruction.model_validate(i)
            for i in copy.deepcopy(self.task["program"]["instructions"])
        ]
        self.history: List[str] = []
        self.step_count = 0
        self.last_action = "none"
        return self._get_obs()

    # ── CFG construction ──────────────────────────────────────────────────────

    def _build_cfg(self) -> Dict[str, Any]:
        n = len(self.program)
        edges: Dict[int, List[int]] = {i: [] for i in range(n)}

        for i, inst in enumerate(self.program):
            op = inst.op  # already lowercased by validator

            if op in {"ret", "return", "stop"}:
                continue

            if op == "jmp" and inst.args and isinstance(inst.args[0], int):
                tgt = inst.args[0]
                if 0 <= tgt < n:
                    edges[i].append(tgt)
                continue

            if op in {"br", "cbr"} and len(inst.args) >= 2:
                t, f = inst.args[0], inst.args[1]
                if isinstance(t, int) and 0 <= t < n:
                    edges[i].append(t)
                if isinstance(f, int) and 0 <= f < n:
                    edges[i].append(f)
                continue

            if i + 1 < n:
                edges[i].append(i + 1)

        exits = [i for i in range(n) if not edges[i]]

        return {
            "entry": 0 if n > 0 else None,
            "edges": edges,
            "exit": exits,
        }

    # ── derived features ──────────────────────────────────────────────────────

    def _op_counts(self) -> Dict[str, int]:
        """Count occurrences of each opcode in the current program."""
        return dict(Counter(inst.op for inst in self.program))

    def _defined_vars(self) -> List[str]:
        """All variable names that appear as the output of some instruction."""
        seen: Dict[str, None] = {}  # ordered-set via insertion-order dict
        for inst in self.program:
            if inst.out and inst.out not in seen:
                seen[inst.out] = None
        return list(seen)


    def _live_vars(self) -> List[str]:
        """
        Variables live at program *entry*, computed by a single-pass
        backward liveness analysis over the linear instruction list.

        A variable is live at entry if it is used before it is defined
        (i.e. it must come from the environment / caller).

        Algorithm (for a straight-line / reducible CFG approximation):
          • Traverse instructions in reverse order.
          • When we see a USE  of variable v  → mark v live.
          • When we see a DEF  of variable v  → remove v from the live set
            (definition kills liveness above this point).
        """
        live: Set[str] = set()
        for inst in reversed(self.program):
            # kill: variable defined here is no longer live above this point
            if inst.out:
                live.discard(inst.out)
            # gen: variables used here become live above this point
            for arg in inst.args:
                if isinstance(arg, str):
                    live.add(arg)
        return sorted(live)

    # ── variable-set helpers (used by _can_apply) ─────────────────────────────

    def _used_vars(self) -> Set[str]:
        used: Set[str] = set()
        for inst in self.program:
            for arg in inst.args:
                if isinstance(arg, str):
                    used.add(arg)
        return used

    def _output_vars(self) -> Set[str]:
        if not self.program:
            return set()
        last = self.program[-1]
        return {last.out} if last.out else set()

    # ── observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> Observation:
        return Observation(
            num_instructions=len(self.program),
            steps_left=self.max_steps - self.step_count,
            last_action=self.last_action,
            program=self.program,
            cfg=self._build_cfg(),
            op_counts=self._op_counts(),
            live_vars=self._live_vars(),
            defined_vars=self._defined_vars(),
        )

    # ── applicability checks ──────────────────────────────────────────────────

    def _can_apply(self, name: str) -> bool:
            if name in {"noop", "stop"}: return True
            if not self.program: return False

            if name == "const_fold":
                const_map = {i.out: i.args[0] for i in self.program if i.op == "const" and i.out}
                for inst in self.program:
                    if inst.op == "const": continue
                    args = [const_map.get(a, a) if isinstance(a, str) else a for a in inst.args]
                    if args and all(isinstance(x, (int, float)) for x in args) and is_math(inst.op):
                        return True
                return False

            if name == "dead_code_elim":
                used = self._used_vars()
                # An instruction is dead if it has an output that is never used and it is pure
                for inst in self.program:
                    if inst.out and inst.out not in used and is_pure(inst.op):
                        return True
                return False
            
            if name == "copy_prop":
                # Can apply if there is an 'id' instruction whose result is used later
                ids = {inst.out for inst in self.program if inst.op == "id"}
                return len(ids.intersection(self._used_vars())) > 0

            if name == "local_cse":
                table = {}
                for inst in self.program:
                    if is_math(inst.op):
                        key = normalize(inst.op, inst.args)
                        if key in table: return True
                        if inst.out: table[key] = inst.out
                return False

            if name == "code_motion":
                for idx, inst in enumerate(self.program):
                    if inst.op in {"jmp", "br", "cbr"}:
                        for arg in inst.args:
                            if isinstance(arg, int) and arg <= idx: # Back-edge found
                                loop_body = self.program[arg : idx+1]
                                defs = {i.out for i in loop_body if i.out}
                                for i in loop_body:
                                    if is_math(i.op) and is_pure(i.op):
                                        if all(a not in defs for a in i.args if isinstance(a, str)):
                                            return True
                return False

            return False

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: Action):
        if action.pass_name not in AVAILABLE_PASSES:
            raise ValueError(f"Unknown pass '{action.pass_name}'")

        prev_len = len(self.program)
        self.step_count += 1
        self.last_action = action.pass_name
        useful = self._can_apply(action.pass_name)

        # 1. Handle STOP action
        if action.pass_name == "stop":
            # Check ALL optimisations, not just two
            no_opts_left = not any(self._can_apply(p) for p in ["const_fold", "dead_code_elim", "code_motion", "copy_prop", "local_cse"])
            reward = 1.0 if no_opts_left else -0.5
            return self._get_obs(), reward, True, {"useful": False}

        # 2. Apply Optimization
        if useful:
            pass_func = AVAILABLE_PASSES[action.pass_name]
            
            # Special case for code_motion which handles List[Instruction]
            if action.pass_name == "code_motion":
                self.program = pass_func(self.program)
            else:
                # Convert List[Instruction] -> Dict for other passes
                prog_dict = {"instructions": [inst.model_dump() for inst in self.program]}
                result_dict = pass_func(prog_dict)
                # Convert Dict -> List[Instruction] back
                self.program = [Instruction.model_validate(i) for i in result_dict["instructions"]]
                
            self.history.append(action.pass_name)

        # 3. Reward Calculation
        removed = prev_len - len(self.program)
        # Give a small penalty for useless steps to prevent infinite loops
        reward = (removed * 1.0) if useful else -0.1
        
        done = self.step_count >= self.max_steps
        return self._get_obs(), reward, done, {"useful": useful}

    # ── external state snapshot ───────────────────────────────────────────────

    def state(self) -> Dict[str, Any]:
        return {
            "num_instructions": len(self.program),
            "history": self.history,
            "cfg": self._build_cfg(),
            "op_counts": self._op_counts(),
            "live_vars": self._live_vars(),
            "defined_vars": self._defined_vars(),
        }