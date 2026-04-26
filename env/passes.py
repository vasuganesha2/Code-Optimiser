from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from env.models import IRProgram

# ──────────────────────────────────────────────────────────────────────────────
# Primitive helpers (duplicated here so passes.py is self-contained)
# ──────────────────────────────────────────────────────────────────────────────

_MATH_OPS: Set[str] = {
    "add", "sub", "mul", "div", "mod", "neg",
    "and", "or", "xor", "not", "shl", "shr",
    "lt", "le", "gt", "ge", "eq", "ne",
}
_TERMINATOR_OPS: Set[str] = {"ret", "return", "stop", "jmp", "br", "cbr"}
_SIDE_EFFECT_OPS: Set[str] = {"store", "print", "call"} | _TERMINATOR_OPS
_COMMUTATIVE_OPS: Set[str] = {"add", "mul", "and", "or", "xor", "eq", "ne"}


def is_math(op: str) -> bool:
    return op in _MATH_OPS


def is_terminator(op: str) -> bool:
    return op in _TERMINATOR_OPS


def is_pure(op: str) -> bool:
    return op not in _SIDE_EFFECT_OPS


def is_constant(op: str, args: List[Any]) -> bool:
    return op == "const" and len(args) == 1 and isinstance(args[0], (int, float))


def can_fold(op: str, args: List[Any]) -> bool:
    return is_math(op) and all(isinstance(a, (int, float)) for a in args)


def evaluate(op: str, args: List[Any]) -> Optional[Any]:
    try:
        a = args[0] if args else None
        b = args[1] if len(args) > 1 else None
        if op == "add":  return a + b
        if op == "sub":  return a - b
        if op == "mul":  return a * b
        if op == "div":  return None if b == 0 else a / b
        if op == "mod":  return None if b == 0 else a % b
        if op == "neg":  return -a
        if op == "and":  return int(a) & int(b)
        if op == "or":   return int(a) | int(b)
        if op == "xor":  return int(a) ^ int(b)
        if op == "not":  return ~int(a)
        if op == "shl":  return int(a) << int(b)
        if op == "shr":  return int(a) >> int(b)
        if op == "lt":   return int(a < b)
        if op == "le":   return int(a <= b)
        if op == "gt":   return int(a > b)
        if op == "ge":   return int(a >= b)
        if op == "eq":   return int(a == b)
        if op == "ne":   return int(a != b)
    except Exception:
        return None
    return None


def normalize(op: str, args: List[Any]) -> Tuple:
    str_args = [str(a) for a in args]
    if op in _COMMUTATIVE_OPS:
        str_args = sorted(str_args)
    return (op, *str_args)


# ──────────────────────────────────────────────────────────────────────────────
# Data-Flow Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_graph(
    instructions: List[Dict[str, Any]]
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    n = len(instructions)
    succs: Dict[int, List[int]] = {i: [] for i in range(n)}
    preds: Dict[int, List[int]] = {i: [] for i in range(n)}

    for i, inst in enumerate(instructions):
        op = inst["op"]
        if op in {"ret", "return", "stop"}:
            continue
        args = inst.get("args", [])
        if op == "jmp" and args and isinstance(args[0], int):
            tgt = args[0]
            if 0 <= tgt < n:
                succs[i].append(tgt)
                preds[tgt].append(i)
            continue
        if op in {"br", "cbr"} and len(args) >= 2:
            for tgt in args[:2]:
                if isinstance(tgt, int) and 0 <= tgt < n:
                    succs[i].append(tgt)
                    preds[tgt].append(i)
            continue
        if i + 1 < n:
            succs[i].append(i + 1)
            preds[i + 1].append(i)
    return succs, preds


def _reaching_definitions(
    instructions: List[Dict[str, Any]],
    succs: Dict[int, List[int]],
    preds: Dict[int, List[int]],
) -> List[Set[Tuple[str, int]]]:
    n = len(instructions)
    IN  = [set() for _ in range(n)]
    OUT = [set() for _ in range(n)]
    GEN = [set() for _ in range(n)]
    KILL = [set() for _ in range(n)]

    def_locs: Dict[str, Set[int]] = {}
    for i, inst in enumerate(instructions):
        v = inst.get("out")
        if v:
            def_locs.setdefault(v, set()).add(i)

    for i, inst in enumerate(instructions):
        v = inst.get("out")
        if v:
            GEN[i].add((v, i))
            KILL[i] = {(v, loc) for loc in def_locs[v] if loc != i}

    changed = True
    while changed:
        changed = False
        for i in range(n):
            new_in: Set = set()
            for p in preds[i]:
                new_in.update(OUT[p])
            IN[i] = new_in
            new_out = GEN[i] | (IN[i] - KILL[i])
            if OUT[i] != new_out:
                OUT[i] = new_out
                changed = True
    return IN


# ──────────────────────────────────────────────────────────────────────────────
# Optimization Passes
# ──────────────────────────────────────────────────────────────────────────────

def loop_invariant_code_motion(program: List[Any]) -> List[Any]:
    """Identifies loops via back-edges and hoists loop-invariant instructions."""
    insts = [i.model_dump() if hasattr(i, "model_dump") else i for i in program]
    if not insts:
        return program

    loops = []
    for idx, inst in enumerate(insts):
        op = inst.get("op")
        args = inst.get("args", [])
        if op in {"jmp", "br", "cbr"}:
            for arg in args:
                if isinstance(arg, int) and arg <= idx:
                    loops.append((arg, idx))

    if not loops:
        return program

    new_insts = list(insts)
    for header, tail in loops:
        loop_indices = list(range(header, tail + 1))
        defs_in_loop = {
            new_insts[i].get("out")
            for i in loop_indices
            if new_insts[i].get("out")
        }

        invariants_to_hoist = []
        for i in loop_indices:
            inst = new_insts[i]
            op, args, out = inst.get("op"), inst.get("args", []), inst.get("out")
            if (
                is_pure(op) and is_math(op) and out
                and all(a not in defs_in_loop for a in args if isinstance(a, str))
            ):
                invariants_to_hoist.append(i)

        current_header = header
        for i in reversed(invariants_to_hoist):
            moving_inst = new_insts.pop(i)
            new_insts.insert(current_header, moving_inst)
            current_header += 1

    from env.models import Instruction
    return [Instruction(**i) for i in new_insts]


def copy_prop(program: Dict[str, Any]) -> Dict[str, Any]:
    """Global Copy Propagation using Reaching Definitions."""
    insts = program.get("instructions", [])
    if not insts:
        return program

    succs, preds = _build_graph(insts)
    IN_sets = _reaching_definitions(insts, succs, preds)

    new_insts = []
    for i, inst in enumerate(insts):
        args = inst.get("args", [])
        new_args = list(args)

        for arg_idx, arg in enumerate(args):
            if isinstance(arg, str):
                reaching_defs = [
                    loc for var, loc in IN_sets[i] if var == arg
                ]
                if reaching_defs:
                    propagate_var = None
                    can_propagate = True
                    for loc in reaching_defs:
                        def_inst = insts[loc]
                        if (
                            def_inst["op"] == "id"
                            and len(def_inst["args"]) == 1
                            and isinstance(def_inst["args"][0], str)
                        ):
                            if propagate_var is None:
                                propagate_var = def_inst["args"][0]
                            elif propagate_var != def_inst["args"][0]:
                                can_propagate = False
                                break
                        else:
                            can_propagate = False
                            break
                    if can_propagate and propagate_var:
                        new_args[arg_idx] = propagate_var

        new_insts.append({**inst, "args": new_args})

    program["instructions"] = new_insts
    return program


def const_fold(program: Dict[str, Any]) -> Dict[str, Any]:
    """Global Constant Folding using Reaching Definitions."""
    insts = program.get("instructions", [])
    if not insts:
        return program

    succs, preds = _build_graph(insts)
    IN_sets = _reaching_definitions(insts, succs, preds)

    new_insts = []
    for i, inst in enumerate(insts):
        op, args, out = inst["op"], inst.get("args", []), inst.get("out")
        new_args = list(args)

        for arg_idx, arg in enumerate(args):
            if isinstance(arg, str):
                reaching_defs = [
                    loc for var, loc in IN_sets[i] if var == arg
                ]
                if reaching_defs:
                    propagate_val = None
                    can_propagate = True
                    for loc in reaching_defs:
                        def_inst = insts[loc]
                        if (
                            def_inst["op"] == "const"
                            and len(def_inst["args"]) == 1
                            and isinstance(def_inst["args"][0], (int, float))
                        ):
                            if propagate_val is None:
                                propagate_val = def_inst["args"][0]
                            elif propagate_val != def_inst["args"][0]:
                                can_propagate = False
                                break
                        else:
                            can_propagate = False
                            break
                    if can_propagate and propagate_val is not None:
                        new_args[arg_idx] = propagate_val

        if can_fold(op, new_args) and out:
            result = evaluate(op, new_args)
            if result is not None:
                new_insts.append({"op": "const", "args": [result], "out": out})
                continue

        new_insts.append({**inst, "args": new_args})

    program["instructions"] = new_insts
    return program


def dead_code_elim(program: Dict[str, Any]) -> Dict[str, Any]:
    """Global Dead Code Elimination using Liveness Analysis."""
    insts = program.get("instructions", [])
    if not insts:
        return program

    n = len(insts)
    succs, preds = _build_graph(insts)
    exit_nodes = {i for i in range(n) if not succs[i]}

    # BUG FIX #3: Seed the final output variable as live at program exits,
    # otherwise the last computed value is incorrectly eliminated.
    final_out: Optional[str] = None
    for inst in reversed(insts):
        if inst.get("out"):
            final_out = inst["out"]
            break

    IN:  List[Set[str]] = [set() for _ in range(n)]
    OUT: List[Set[str]] = [set() for _ in range(n)]

    changed = True
    while changed:
        changed = False
        for i in reversed(range(n)):
            inst = insts[i]

            new_out: Set[str] = set()
            for s in succs[i]:
                new_out.update(IN[s])
            # Seed exits with the program's final output
            if i in exit_nodes and final_out:
                new_out.add(final_out)
            OUT[i] = new_out

            new_in = set(OUT[i])
            out_var = inst.get("out")
            if out_var:
                new_in.discard(out_var)
            for arg in inst.get("args", []):
                if isinstance(arg, str):
                    new_in.add(arg)

            if IN[i] != new_in:
                IN[i] = new_in
                changed = True

    new_insts = []
    for i, inst in enumerate(insts):
        op, out = inst["op"], inst.get("out")
        if is_terminator(op) or not is_pure(op) or (out and out in OUT[i]):
            new_insts.append(inst)

    program["instructions"] = new_insts
    return program


# ── CSE helpers ───────────────────────────────────────────────────────────────

def _available_expressions(
    instructions: List[Dict[str, Any]],
    succs: Dict[int, List[int]],
    preds: Dict[int, List[int]],
) -> List[Dict[Any, str]]:
    """
    Computes Global Available Expressions (intersection-based).
    Returns, for each instruction index i, a dict mapping expression_key
    to the variable that holds its result at the entry of i.
    """
    n = len(instructions)
    all_exprs: Dict[Tuple, Set[int]] = {}
    for i, inst in enumerate(instructions):
        if is_math(inst["op"]) and is_pure(inst["op"]):
            key = normalize(inst["op"], inst.get("args", []))
            all_exprs.setdefault(key, set()).add(i)

    GEN: List[Optional[Tuple]] = [None] * n
    KILL: List[Set[Tuple]] = [set() for _ in range(n)]

    for i, inst in enumerate(instructions):
        if is_math(inst["op"]) and is_pure(inst["op"]):
            GEN[i] = normalize(inst["op"], inst.get("args", []))

        out_var = inst.get("out")
        if out_var:
            for expr_key in all_exprs:
                # BUG FIX #1: expr_key[1] is a single string — using `in` on a
                # string gives a substring match, not element membership.
                # expr_key[1:] is a tuple, so `in` correctly tests membership.
                if out_var in expr_key[1:]:
                    KILL[i].add(expr_key)

    universe = set(all_exprs.keys())
    OUT = [set(universe) for _ in range(n)]
    if n > 0:
        OUT[0] = set()

    changed = True
    while changed:
        changed = False
        for i in range(n):
            if not preds[i]:
                new_in: Set = set()
            else:
                new_in = set.intersection(*(OUT[p] for p in preds[i]))

            temp_out = new_in - KILL[i]
            if GEN[i]:
                temp_out.add(GEN[i])

            if OUT[i] != temp_out:
                OUT[i] = temp_out
                changed = True

    results: List[Dict[Any, str]] = []
    for i in range(n):
        expr_to_var: Dict[Any, str] = {}
        for expr in OUT[i]:
            for loc in all_exprs[expr]:
                if loc < i:
                    expr_to_var[expr] = instructions[loc]["out"]
        results.append(expr_to_var)
    return results


def local_cse(program: Dict[str, Any]) -> Dict[str, Any]:
    """Global Common Subexpression Elimination (via available-expressions analysis)."""
    insts = program.get("instructions", [])
    if not insts:
        return program

    succs, preds = _build_graph(insts)
    available_in = _available_expressions(insts, succs, preds)

    new_insts = []
    for i, inst in enumerate(insts):
        op, args, out = inst["op"], inst.get("args", []), inst.get("out")

        if is_math(op) and is_pure(op):
            expr_key = normalize(op, args)
            if expr_key in available_in[i]:
                prev_var = available_in[i][expr_key]
                new_insts.append({"op": "id", "args": [prev_var], "out": out})
                continue

        new_insts.append(inst)

    program["instructions"] = new_insts
    return program


# BUG FIX #4a: global_cse is imported by env.py but was never defined.
# local_cse was already upgraded to a global analysis, so we simply alias it.
global_cse = local_cse


# ── Store / Load passes ───────────────────────────────────────────────────────

def store_load_fwd(program: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store-to-Load Forwarding.
    When a load reads from a pointer that was most recently stored to (with no
    intervening aliasing stores), replace the load with an `id` of the stored value.
    """
    insts = program.get("instructions", [])
    if not insts:
        return program

    store_map: Dict[Any, Any] = {}   # ptr -> value-variable
    new_insts = []

    for inst in insts:
        op, args, out = inst["op"], inst.get("args", []), inst.get("out")

        if op == "store" and len(args) == 2:
            val, ptr = args[0], args[1]
            store_map[ptr] = val
            new_insts.append(inst)
        elif op == "load" and len(args) == 1:
            ptr = args[0]
            if ptr in store_map:
                # Forward stored value; the load instruction is replaced.
                new_insts.append({"op": "id", "args": [store_map[ptr]], "out": out})
            else:
                new_insts.append(inst)
        else:
            # Conservative: any unrecognised store clears the cache entirely.
            if op == "store":
                store_map.clear()
            new_insts.append(inst)

    program["instructions"] = new_insts
    return program


def dead_store_elim(program: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dead Store Elimination.
    A store to pointer P is dead if P is stored to again before any intervening
    load from P.  We make a single forward pass tracking the last store index
    for each pointer and mark the earlier one dead when a second store is seen.
    """
    insts = program.get("instructions", [])
    if not insts:
        return program

    last_store: Dict[Any, int] = {}   # ptr -> instruction index of last store
    dead: Set[int] = set()

    for i, inst in enumerate(insts):
        op, args = inst["op"], inst.get("args", [])

        if op == "load" and args:
            # A load keeps the previous store alive.
            last_store.pop(args[0], None)
        elif op == "store" and len(args) == 2:
            _val, ptr = args[0], args[1]
            if ptr in last_store:
                dead.add(last_store[ptr])   # previous store to same ptr is dead
            last_store[ptr] = i

    new_insts = [inst for i, inst in enumerate(insts) if i not in dead]
    program["instructions"] = new_insts
    return program


# ── Lazy Code Motion (Algorithm 9.36) ────────────────────────────────────────

def _partial_redundancy_elimination(program: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full implementation of Algorithm 9.36: Lazy Code Motion.
    Operates on a flat instruction dict (same interface as the other passes).
    """
    insts = program.get("instructions", [])
    if not insts:
        return program
    n = len(insts)
    succs, preds = _build_graph(insts)

    # Step 1: Collect universe U, e_use, e_kill
    U: Set[Tuple] = set()
    e_use:  List[Set[Tuple]] = [set() for _ in range(n)]
    e_kill: List[Set[Tuple]] = [set() for _ in range(n)]

    for i, inst in enumerate(insts):
        op, args, out = inst["op"], inst.get("args", []), inst.get("out")
        if is_math(op) and is_pure(op):
            expr = normalize(op, args)
            U.add(expr)
            e_use[i].add(expr)
        if out:
            for expr in U:
                # BUG FIX #2: same substring-vs-membership bug as in
                # _available_expressions.  Use expr[1:] (tuple slice).
                if out in expr[1:]:
                    e_kill[i].add(expr)

    def solve(
        forward: bool,
        intersect: bool,
        transfer_func,
        initial_set: Set,
    ) -> Tuple[List[Set], List[Set]]:
        nodes = list(range(n))
        if not forward:
            nodes.reverse()

        IN  = [set(initial_set) for _ in range(n)]
        OUT = [set(initial_set) for _ in range(n)]

        if forward:
            IN[0] = set()
        else:
            OUT[n - 1] = set()

        changed = True
        while changed:
            changed = False
            for i in nodes:
                neighbors = preds[i] if forward else succs[i]
                if neighbors:
                    combined = set(OUT[neighbors[0]] if forward else IN[neighbors[0]])
                    for nb in neighbors[1:]:
                        nb_set = OUT[nb] if forward else IN[nb]
                        if intersect:
                            combined &= nb_set
                        else:
                            combined |= nb_set
                else:
                    combined = set()

                if forward:
                    IN[i] = combined
                else:
                    OUT[i] = combined

                new_val = transfer_func(i, IN[i] if forward else OUT[i])
                target = OUT if forward else IN
                if target[i] != new_val:
                    target[i] = new_val
                    changed = True
        return IN, OUT

    # Step 2: Anticipated (backward, intersect)
    antic_in, _ = solve(
        False, True,
        lambda i, out_i: e_use[i] | (out_i - e_kill[i]),
        U,
    )

    # Step 3: Available (forward, intersect)
    avail_in, _ = solve(
        True, True,
        lambda i, in_i: (antic_in[i] | in_i) - e_kill[i],
        U,
    )

    # Step 4: Earliest
    earliest = [antic_in[i] - avail_in[i] for i in range(n)]

    # Step 5: Postponable (forward, intersect)
    post_in, _ = solve(
        True, True,
        lambda i, in_i: (earliest[i] | in_i) - e_use[i],
        U,
    )

    # Step 6: Latest
    latest = [set() for _ in range(n)]
    for i in range(n):
        ear_post = earliest[i] | post_in[i]
        succ_comp = set(U)
        if succs[i]:
            for s in succs[i]:
                succ_comp &= (earliest[s] | post_in[s])
        else:
            succ_comp = set()
        latest[i] = ear_post & (e_use[i] | (U - succ_comp))

    # Step 7: Used (backward, union)
    _, used_out = solve(
        False, False,
        lambda i, out_i: (e_use[i] | out_i) - latest[i],
        set(),
    )

    # Step 8: Transform
    expr_to_temp = {expr: f"t_lcm_{idx}" for idx, expr in enumerate(U)}
    new_insts = []
    for i, inst in enumerate(insts):
        for expr in latest[i] & used_out[i]:
            new_insts.append({
                "op": expr[0],
                "args": list(expr[1:]),
                "out": expr_to_temp[expr],
            })

        op, args, out = inst["op"], inst.get("args", []), inst.get("out")
        if is_math(op) and is_pure(op):
            expr = normalize(op, args)
            if expr in (used_out[i] | latest[i]):
                new_insts.append({"op": "id", "args": [expr_to_temp[expr]], "out": out})
                continue

        new_insts.append(inst)

    program["instructions"] = new_insts
    return program


def lazy_code_motion(ir: "IRProgram") -> "IRProgram":
    """
    BUG FIX #5: env.py calls this function with an IRProgram (it uses the
    flat↔IR bridge for the 'lcm' pass).  The old codebase had no such function
    — only `partial_redundancy_elimination` operating on plain dicts.

    This wrapper flattens the IR, runs PRE, then reconstructs the IR.
    """
    from env.models import IRProgram as IR, BasicBlock

    flat_insts: List[Dict[str, Any]] = []
    for block in ir.blocks:
        flat_insts.extend(block.insts)

    if not flat_insts:
        return ir

    result = _partial_redundancy_elimination({"instructions": flat_insts})
    new_insts = result["instructions"]

    # Re-partition into basic blocks on terminator boundaries.
    new_blocks: List[BasicBlock] = []
    current: List[Dict[str, Any]] = []
    block_idx = 0
    for inst in new_insts:
        current.append(inst)
        if inst.get("op") in _TERMINATOR_OPS:
            new_blocks.append(
                BasicBlock(label=f"b{block_idx}", insts=current, preds=[], succs=[])
            )
            current = []
            block_idx += 1
    if current:
        new_blocks.append(
            BasicBlock(label=f"b{block_idx}", insts=current, preds=[], succs=[])
        )

    return IR(blocks=new_blocks, entry="b0")


# ── Misc ──────────────────────────────────────────────────────────────────────

def noop(program: Dict[str, Any]) -> Dict[str, Any]:
    return program