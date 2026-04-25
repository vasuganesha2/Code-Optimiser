import copy
from collections import Counter
from typing import Any, Dict, List, Set, Union, Tuple

from env.models import Instruction, Observation, Action, IRProgram, BasicBlock
from env.passes import (
    const_fold, 
    dead_code_elim, 
    noop, 
    loop_invariant_code_motion,
    copy_prop,
    local_cse,
    global_cse,
    lazy_code_motion # Algorithm 9.36
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
    "global_cse": global_cse,
    "lcm": lazy_code_motion,
    "stop": noop,
}

class CompilerEnv:
    def __init__(self, task: Dict[str, Any]):
        self.task = task
        self.max_steps = 20
        self.reset()

    def reset(self) -> Observation:
        self.program: List[Instruction] = [
            Instruction.model_validate(i)
            for i in copy.deepcopy(self.task["program"]["instructions"])
        ]
        self.history: List[str] = []
        self.step_count = 0
        self.last_action = "none"
        return self._get_obs()

    # ── IR Translation ────────────────────────────────────────────────────────

    def _to_ir(self) -> IRProgram:
        """Converts flat instructions into a basic-block structured IRProgram."""
        if not self.program:
            return IRProgram(blocks=[], entry="")
            
        # Basic partitioning logic
        blocks_data = []
        current_insts = []
        block_count = 0
        
        for inst in self.program:
            current_insts.append(inst.model_dump())
            if is_terminator(inst.op):
                blocks_data.append(BasicBlock(
                    label=f"b{block_count}",
                    insts=current_insts,
                    preds=[], succs=[] # These are filled by LCM internal logic or a separate builder
                ))
                current_insts = []
                block_count += 1
        
        if current_insts:
            blocks_data.append(BasicBlock(label=f"b{block_count}", insts=current_insts, preds=[], succs=[]))
            
        return IRProgram(blocks=blocks_data, entry="b0")

    def _from_ir(self, ir: IRProgram) -> List[Instruction]:
        """Flattens an IRProgram back into a linear instruction list."""
        flat = []
        for block in ir.blocks:
            for inst_dict in block.insts:
                flat.append(Instruction.model_validate(inst_dict))
        return flat

    # ── CFG & Liveness ────────────────────────────────────────────────────────

    def _build_cfg(self) -> Dict[str, Any]:
        n = len(self.program)
        edges: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i, inst in enumerate(self.program):
            op = inst.op
            if op in {"ret", "return", "stop"}: continue
            if op == "jmp" and inst.args and isinstance(inst.args[0], int):
                tgt = inst.args[0]
                if 0 <= tgt < n: edges[i].append(tgt)
            elif op in {"br", "cbr"} and len(inst.args) >= 2:
                for tgt in inst.args[:2]:
                    if isinstance(tgt, int) and 0 <= tgt < n: edges[i].append(tgt)
            elif i + 1 < n:
                edges[i].append(i + 1)
        return {"entry": 0 if n > 0 else None, "edges": edges, "exit": [i for i in range(n) if not edges[i]]}

    def _live_at_exit(self) -> List[Set[str]]:
        """Global backward liveness analysis."""
        n = len(self.program)
        cfg = self._build_cfg()["edges"]
        in_sets = [set() for _ in range(n)]
        out_sets = [set() for _ in range(n)]
        
        changed = True
        while changed:
            changed = False
            for i in reversed(range(n)):
                inst = self.program[i]
                new_out = set().union(*(in_sets[s] for s in cfg[i]))
                new_in = set(new_out)
                if inst.out: new_in.discard(inst.out)
                for arg in inst.args:
                    if isinstance(arg, str): new_in.add(arg)
                
                if new_in != in_sets[i] or new_out != out_sets[i]:
                    in_sets[i], out_sets[i] = new_in, new_out
                    changed = True
        return out_sets

    # ── Applicability & Metrics ───────────────────────────────────────────────

    def _can_apply(self, name: str) -> bool:
        if name in {"noop", "stop"}: return True
        if not self.program: return False

        if name == "dead_code_elim":
            out_sets = self._live_at_exit()
            for i, inst in enumerate(self.program):
                if inst.out and inst.out not in out_sets[i] and is_pure(inst.op):
                    return True
            return False

        if name == "local_cse" or name == "lcm":
            seen = set()
            for inst in self.program:
                if is_math(inst.op):
                    key = normalize(inst.op, inst.args)
                    if key in seen: return True
                    seen.add(key)
            return False

        if name == "const_fold":
            # Simplified: checks if any math op has a constant argument
            consts = {i.out for i in self.program if i.op == "const"}
            for inst in self.program:
                if is_math(inst.op) and any(a in consts for a in inst.args if isinstance(a, str)):
                    return True
            return False

        if name == "copy_prop":
            ids = {inst.out for inst in self.program if inst.op == "id"}
            used = {a for i in self.program for a in i.args if isinstance(a, str)}
            return not ids.isdisjoint(used)

        if name == "code_motion":
            # Check for instructions where arguments are not defined in the same potential loop body
            return True # Conservative guess for RL agents

        return False

    def _get_obs(self) -> Observation:
        return Observation(
            num_instructions=len(self.program),
            steps_left=self.max_steps - self.step_count,
            last_action=self.last_action,
            program=self.program,
            cfg=self._build_cfg(),
            op_counts=dict(Counter(inst.op for inst in self.program)),
            live_vars=sorted(list(self._live_at_exit()[0])) if self.program else [],
            defined_vars=list({inst.out for inst in self.program if inst.out}),
        )

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        if action.pass_name not in AVAILABLE_PASSES:
            raise ValueError(f"Unknown pass '{action.pass_name}'")

        prev_len = len(self.program)
        self.step_count += 1
        self.last_action = action.pass_name
        useful = self._can_apply(action.pass_name)

        if action.pass_name == "stop":
            rem_opts = any(self._can_apply(p) for p in ["const_fold", "dead_code_elim", "code_motion", "copy_prop", "local_cse", "lcm"])
            reward = 2.0 if not rem_opts else -1.0
            return self._get_obs(), reward, True, {"useful": False}

        if useful:
            pass_func = AVAILABLE_PASSES[action.pass_name]
            
            if action.pass_name == "lcm":
                # Bridge: Flat -> IR -> Flat
                ir_prog = self._to_ir()
                optimized_ir = pass_func(ir_prog)
                self.program = self._from_ir(optimized_ir)
            elif action.pass_name == "code_motion":
                self.program = pass_func(self.program)
            else:
                # Standard Dict-based interface
                prog_dict = {"instructions": [inst.model_dump() for inst in self.program]}
                result = pass_func(prog_dict)
                self.program = [Instruction.model_validate(i) for i in result["instructions"]]
                
            self.history.append(action.pass_name)

        reduction = prev_len - len(self.program)
        reward = (reduction * 1.5) if useful else -0.1
        done = self.step_count >= self.max_steps
        
        return self._get_obs(), reward, done, {"useful": useful}

    def state(self) -> Dict[str, Any]:
        return {
            "num_instructions": len(self.program),
            "history": self.history,
            "cfg": self._build_cfg(),
            "op_counts": dict(Counter(inst.op for inst in self.program)),
            "live_vars": sorted(list(self._live_at_exit()[0])) if self.program else [],
            "defined_vars": list({inst.out for inst in self.program if inst.out}),
        }