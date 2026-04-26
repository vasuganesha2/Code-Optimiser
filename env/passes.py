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
    """
    Flat-list LICM heuristic:
    - detect back-edges
    - find pure math instructions in the loop whose operands are not defined in the loop
    - move them before the loop header
    """
    insts = [
        i.model_dump() if hasattr(i, "model_dump") else dict(i)
        for i in program
    ]
    if not insts:
        return program

    # Collect back-edges as (header, tail)
    loops = []
    for idx, inst in enumerate(insts):
        op = inst.get("op")
        args = inst.get("args", [])
        if op == "jmp":
            if args and isinstance(args[0], int) and args[0] < idx:
                loops.append((args[0], idx))
        elif op in {"br", "cbr"}:
            for tgt in args[:2]:
                if isinstance(tgt, int) and tgt < idx:
                    loops.append((tgt, idx))

    if not loops:
        return program

    # Process inner/later loops first so index shifts do not break earlier ones.
    loops = sorted(set(loops), key=lambda x: (x[0], x[1]), reverse=True)

    for header, tail in loops:
        if header < 0 or tail >= len(insts) or header >= tail:
            continue

        loop_indices = list(range(header, tail + 1))

        # Values defined anywhere inside the loop
        defs_in_loop = {
            insts[i].get("out")
            for i in loop_indices
            if insts[i].get("out") is not None
        }

        movable_indices = []
        for i in loop_indices:
            inst = insts[i]
            op = inst.get("op")
            args = inst.get("args", [])
            out = inst.get("out")

            # Do not move control flow or side-effecting ops
            if not out:
                continue
            if not is_pure(op) or not is_math(op):
                continue
            if is_terminator(op):
                continue

            # Only hoist if every variable operand comes from outside the loop
            invariant = True
            for a in args:
                if isinstance(a, str) and a in defs_in_loop:
                    invariant = False
                    break

            if invariant:
                movable_indices.append(i)

        if not movable_indices:
            continue

        movable_set = set(movable_indices)

        # Preserve original order of hoisted instructions
        hoisted = [insts[i] for i in movable_indices]
        remaining = [inst for i, inst in enumerate(insts) if i not in movable_set]

        # Insert hoisted instructions just before the original loop header
        insert_pos = sum(1 for i in range(header) if i not in movable_set)
        insts = remaining[:insert_pos] + hoisted + remaining[insert_pos:]

    from env.models import Instruction
    return [Instruction(**i) for i in insts]


# Make sure the pipeline can find it
code_motion = loop_invariant_code_motion


def copy_prop(program: Dict[str, Any]) -> Dict[str, Any]:
    """
    Global copy propagation using reaching definitions.

    Safe rule:
    replace a variable use v with source s only when all reaching
    definitions of v are copies of the same source s.
    """
    insts = program.get("instructions", [])
    if not insts:
        return program

    succs, preds = _build_graph(insts)
    in_sets = _reaching_definitions(insts, succs, preds)

    new_insts: List[Dict[str, Any]] = []

    for i, inst in enumerate(insts):
        args = list(inst.get("args", []))
        new_args = args[:]

        for j, arg in enumerate(args):
            if not isinstance(arg, str):
                continue

            reaching_defs = [loc for var, loc in in_sets[i] if var == arg]
            if not reaching_defs:
                continue

            source_var: Optional[str] = None
            safe = True

            for loc in reaching_defs:
                def_inst = insts[loc]

                # Only propagate through `id x -> y`
                if def_inst.get("op") != "id":
                    safe = False
                    break

                def_args = def_inst.get("args", [])
                if len(def_args) != 1 or not isinstance(def_args[0], str):
                    safe = False
                    break

                src = def_args[0]
                if source_var is None:
                    source_var = src
                elif source_var != src:
                    safe = False
                    break

            if safe and source_var is not None:
                new_args[j] = source_var

        new_insts.append({**inst, "args": new_args})

    program["instructions"] = new_insts
    return program

# copy_propagation = copy_prop


def const_fold(program: Dict[str, Any]) -> Dict[str, Any]:
    """Global constant folding using reaching definitions."""
    insts = program.get("instructions", [])
    if not insts:
        return program

    succs, preds = _build_graph(insts)
    in_sets = _reaching_definitions(insts, succs, preds)

    def get_reaching_const(var: str, i: int) -> Optional[float]:
        reaching_defs = [loc for v, loc in in_sets[i] if v == var]
        if not reaching_defs:
            return None

        value: Optional[float] = None
        for loc in reaching_defs:
            def_inst = insts[loc]
            if def_inst.get("op") != "const":
                return None
            def_args = def_inst.get("args", [])
            if len(def_args) != 1 or not isinstance(def_args[0], (int, float)):
                return None

            v = def_args[0]
            if value is None:
                value = v
            elif value != v:
                return None

        return value

    new_insts: List[Dict[str, Any]] = []

    for i, inst in enumerate(insts):
        op = inst.get("op")
        args = list(inst.get("args", []))
        out = inst.get("out")

        new_args = list(args)
        for j, arg in enumerate(args):
            if isinstance(arg, str):
                const_val = get_reaching_const(arg, i)
                if const_val is not None:
                    new_args[j] = const_val

        if out is not None and can_fold(op, new_args):
            result = evaluate(op, new_args)
            if result is not None:
                new_insts.append({"op": "const", "args": [result], "out": out})
                continue

        new_insts.append({**inst, "args": new_args})

    program["instructions"] = new_insts
    return program

def dead_code_elim(program: Dict[str, Any]) -> Dict[str, Any]:
    """Global dead code elimination using backward liveness analysis."""
    insts = program.get("instructions", [])
    if not insts:
        return program

    n = len(insts)
    succs, preds = _build_graph(insts)

    IN: List[Set[str]] = [set() for _ in range(n)]
    OUT: List[Set[str]] = [set() for _ in range(n)]

    changed = True
    while changed:
        changed = False

        for i in reversed(range(n)):
            inst = insts[i]

            new_out: Set[str] = set()
            for s in succs[i]:
                new_out.update(IN[s])

            new_in = set(new_out)

            out_var = inst.get("out")
            if out_var is not None:
                new_in.discard(out_var)

            for arg in inst.get("args", []):
                if isinstance(arg, str):
                    new_in.add(arg)

            if new_in != IN[i] or new_out != OUT[i]:
                IN[i] = new_in
                OUT[i] = new_out
                changed = True

    new_insts = []
    for i, inst in enumerate(insts):
        op = inst["op"]
        out = inst.get("out")

        # Keep side-effecting ops and terminators.
        # Keep pure ops only if their result is live.
        if is_terminator(op) or not is_pure(op) or (out is not None and out in OUT[i]):
            new_insts.append(inst)

    program["instructions"] = new_insts
    return program



# ── CSE helpers ───────────────────────────────────────────────────────────────
def _available_expressions(
    instructions: List[Dict[str, Any]],
    succs: Dict[int, List[int]],
    preds: Dict[int, List[int]],
) -> List[Dict[Any, str]]:
    n = len(instructions)

    # Collect all expressions
    all_exprs: Dict[Tuple, Set[int]] = {}
    for i, inst in enumerate(instructions):
        if is_math(inst["op"]) and is_pure(inst["op"]):
            key = normalize(inst["op"], inst.get("args", []))
            all_exprs.setdefault(key, set()).add(i)

    # GEN / KILL
    GEN = [set() for _ in range(n)]
    KILL = [set() for _ in range(n)]

    for i, inst in enumerate(instructions):
        if is_math(inst["op"]) and is_pure(inst["op"]):
            key = normalize(inst["op"], inst.get("args", []))
            GEN[i].add(key)

        out_var = inst.get("out")
        if out_var:
            for expr in all_exprs:
                if out_var in expr[1:]:
                    KILL[i].add(expr)

    universe = set(all_exprs.keys())

    IN = [set(universe) for _ in range(n)]
    OUT = [set() for _ in range(n)]

    # Entry node has empty IN
    if n > 0:
        IN[0] = set()

    changed = True
    while changed:
        changed = False
        for i in range(n):
            # IN[i] = intersection of OUT[pred]
            if preds[i]:
                new_in = set.intersection(*(OUT[p] for p in preds[i]))
            else:
                new_in = set()

            # OUT[i] = GEN[i] ∪ (IN[i] - KILL[i])
            new_out = GEN[i] | (new_in - KILL[i])

            if new_in != IN[i] or new_out != OUT[i]:
                IN[i] = new_in
                OUT[i] = new_out
                changed = True

    # Build mapping: expression → variable (ONLY from dominating defs)
    results: List[Dict[Any, str]] = []

    for i in range(n):
        expr_to_var: Dict[Any, str] = {}

        for expr in IN[i]:  # IMPORTANT: use IN, not OUT
            # find nearest dominating definition
            for j in reversed(range(i)):
                inst = instructions[j]
                if (
                    is_math(inst["op"])
                    and is_pure(inst["op"])
                    and normalize(inst["op"], inst.get("args", [])) == expr
                ):
                    expr_to_var[expr] = inst["out"]
                    break

        results.append(expr_to_var)

    return results


def local_cse(program: Dict[str, Any]) -> Dict[str, Any]:
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

                # avoid self-replacement
                if prev_var != out:
                    new_insts.append({
                        "op": "id",
                        "args": [prev_var],
                        "out": out
                    })
                    continue

        new_insts.append(inst)

    program["instructions"] = new_insts
    return program


# local_cse already implements a global analysis; alias for compatibility
global_cse = local_cse


# ── Store / Load passes ───────────────────────────────────────────────────────
def store_load_fwd(program: Dict[str, Any]) -> Dict[str, Any]:
    insts = program.get("instructions", [])
    if not insts:
        return program

    n = len(insts)
    succs, preds = _build_graph(insts)

    # Dataflow: map ptr -> value
    IN = [dict() for _ in range(n)]
    OUT = [dict() for _ in range(n)]

    def meet(maps):
        """Intersection: keep only same ptr->val in all predecessors"""
        if not maps:
            return {}
        keys = set(maps[0].keys())
        for m in maps[1:]:
            keys &= set(m.keys())

        result = {}
        for k in keys:
            vals = {m[k] for m in maps}
            if len(vals) == 1:
                result[k] = vals.pop()
        return result

    changed = True
    while changed:
        changed = False
        for i in range(n):
            # IN = meet of predecessors
            new_in = meet([OUT[p] for p in preds[i]]) if preds[i] else {}

            # transfer
            new_out = dict(new_in)
            inst = insts[i]
            op, args = inst["op"], inst.get("args", [])

            if op == "store" and len(args) == 2:
                val, ptr = args
                new_out[ptr] = val

            elif op == "load":
                pass  # no change

            else:
                # unknown op → may touch memory
                new_out.clear()

            if new_in != IN[i] or new_out != OUT[i]:
                IN[i] = new_in
                OUT[i] = new_out
                changed = True

    # Rewrite
    new_insts = []
    for i, inst in enumerate(insts):
        op, args, out = inst["op"], inst.get("args", []), inst.get("out")

        if op == "load" and len(args) == 1:
            ptr = args[0]
            if ptr in IN[i]:
                new_insts.append({
                    "op": "id",
                    "args": [IN[i][ptr]],
                    "out": out
                })
                continue

        new_insts.append(inst)

    program["instructions"] = new_insts
    return program


def dead_store_elim(program: Dict[str, Any]) -> Dict[str, Any]:
    insts = program.get("instructions", [])
    if not insts:
        return program

    n = len(insts)
    succs, preds = _build_graph(insts)

    # LIVE[i] = set of pointers whose value may be used after i
    IN = [set() for _ in range(n)]
    OUT = [set() for _ in range(n)]

    changed = True
    while changed:
        changed = False
        for i in reversed(range(n)):
            # OUT = union of successors
            new_out = set()
            for s in succs[i]:
                new_out |= IN[s]

            inst = insts[i]
            op, args = inst["op"], inst.get("args", [])

            new_in = set(new_out)

            if op == "load" and args:
                new_in.add(args[0])  # ptr becomes live

            elif op == "store" and len(args) == 2:
                _, ptr = args
                # store kills previous value
                new_in.discard(ptr)

            else:
                # unknown op → assume all memory may be used
                new_in = set(new_out)

            if new_in != IN[i] or new_out != OUT[i]:
                IN[i] = new_in
                OUT[i] = new_out
                changed = True

    # Remove dead stores
    new_insts = []
    for i, inst in enumerate(insts):
        op, args = inst["op"], inst.get("args", [])

        if op == "store" and len(args) == 2:
            _, ptr = args
            if ptr not in OUT[i]:
                continue  # dead store → remove

        new_insts.append(inst)

    program["instructions"] = new_insts
    return program


# ── Lazy Code Motion (Algorithm 9.36) ────────────────────────────────────────
def _partial_redundancy_elimination(program: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full implementation of Algorithm 9.36: Lazy Code Motion.
    Operates on a flat instruction list.
    """
    insts = program.get("instructions", [])
    if not insts:
        return program
    n = len(insts)
    succs, preds = _build_graph(insts)

    # Step 1: Collect universe U, e_use, e_kill
    U: Set[Tuple] = set()
    e_use: List[Set[Tuple]] = [set() for _ in range(n)]
    e_kill: List[Set[Tuple]] = [set() for _ in range(n)]
    for i, inst in enumerate(insts):
        op, args, out = inst["op"], inst.get("args", []), inst.get("out")
        if is_math(op) and is_pure(op):
            expr = normalize(op, args)
            U.add(expr)
            e_use[i].add(expr)
    for i, inst in enumerate(insts):
        out = inst.get("out")
        if out:
            for expr in U:
                if out in expr[1:]:
                    e_kill[i].add(expr)

    def solve_dataflow(forward: bool, meet_is_intersect: bool, transfer_func, init_val):
        IN = [set(init_val) for _ in range(n)]
        OUT = [set(init_val) for _ in range(n)]
        if forward and n > 0:
            IN[0] = set()
        if not forward:
            for i in range(n):
                if not succs[i]:
                    OUT[i] = set()
        changed = True
        while changed:
            changed = False
            order = range(n) if forward else reversed(range(n))
            for i in order:
                if forward and i == 0:
                    continue
                neighbors = preds[i] if forward else succs[i]
                if not neighbors:
                    meet_set = set()
                else:
                    neighbor_states = [OUT[p] if forward else IN[p] for p in neighbors]
                    meet_set = set(neighbor_states[0])
                    for state in neighbor_states[1:]:
                        if meet_is_intersect:
                            meet_set &= state
                        else:
                            meet_set |= state
                if forward:
                    IN[i] = meet_set
                else:
                    OUT[i] = meet_set
                new_state = transfer_func(i, IN[i] if forward else OUT[i])
                target_list = OUT if forward else IN
                if target_list[i] != new_state:
                    target_list[i] = new_state
                    changed = True
        return IN, OUT

    # Step 2: Anticipated Expressions (Backward, Intersect)
    antic_in, antic_out = solve_dataflow(
        forward=False, meet_is_intersect=True, init_val=U,
        transfer_func=lambda i, out_i: e_use[i] | (out_i - e_kill[i])
    )

    # Step 3: Available Expressions (Forward, Intersect)
    avail_in, avail_out = solve_dataflow(
        forward=True, meet_is_intersect=True, init_val=U,
        transfer_func=lambda i, in_i: (antic_in[i] | in_i) - e_kill[i]
    )

    # Step 4: Earliest Placements
    earliest = [antic_in[i] - avail_in[i] for i in range(n)]

    # Step 5: Postponable Expressions (Forward, Intersect)
    postp_in, postp_out = solve_dataflow(
        forward=True, meet_is_intersect=True, init_val=U,
        transfer_func=lambda i, in_i: (earliest[i] | in_i) - e_use[i]
    )

    # Step 6: Latest Placements
    latest = [set() for _ in range(n)]
    for i in range(n):
        base_set = earliest[i] | postp_in[i]
        succ_intersect = set(U)
        if succs[i]:
            for s in succs[i]:
                succ_intersect &= (earliest[s] | postp_in[s])
        else:
            succ_intersect = set()
        latest[i] = base_set & (e_use[i] | (U - succ_intersect))

    # Step 7: Used Expressions (Backward, Union)
    used_in, used_out = solve_dataflow(
        forward=False, meet_is_intersect=False, init_val=set(),
        transfer_func=lambda i, out_i: (e_use[i] | out_i) - latest[i]
    )

    # Step 8: Code Rewriting
    temp_counter = 0
    expr_to_temp = {}
    new_insts = []
    index_shift = 0
    old_to_new_mapping = {}

    for i, inst in enumerate(insts):
        old_to_new_mapping[i] = i + index_shift

        insertions = latest[i] & used_out[i]
        for expr in insertions:
            if expr not in expr_to_temp:
                expr_to_temp[expr] = f"pre_t{temp_counter}"
                temp_counter += 1
            op, args = expr[0], list(expr[1:])
            new_insts.append({"op": op, "args": args, "out": expr_to_temp[expr]})
            index_shift += 1

        replacements = e_use[i] & ((U - latest[i]) | used_out[i])
        op, args, out = inst.get("op"), inst.get("args", []), inst.get("out")

        if is_math(op) and is_pure(op):
            expr = normalize(op, args)
            if expr in replacements and expr in expr_to_temp:
                new_insts.append({"op": "id", "args": [expr_to_temp[expr]], "out": out})
                continue

        new_insts.append(inst)

    # Update control flow targets after insertions shifted indices
    for inst in new_insts:
        op = inst["op"]
        if op == "jmp" and inst.get("args"):
            inst["args"][0] = old_to_new_mapping.get(inst["args"][0], inst["args"][0])
        elif op in {"br", "cbr"} and len(inst.get("args", [])) >= 2:
            inst["args"][0] = old_to_new_mapping.get(inst["args"][0], inst["args"][0])
            inst["args"][1] = old_to_new_mapping.get(inst["args"][1], inst["args"][1])

    program["instructions"] = new_insts
    return program


# FIX: removed dead assignment `lazy_code_motion = _partial_redundancy_elimination`
# that was immediately overwritten by the function definition below.
def lazy_code_motion(ir: "IRProgram") -> "IRProgram":
    """
    env.py calls this with an IRProgram (flat<->IR bridge for the 'lcm' pass).
    Flattens the IR, runs PRE, then reconstructs the IR.
    """
    from env.models import IRProgram as IR, BasicBlock
    flat_insts: List[Dict[str, Any]] = []
    for block in ir.blocks:
        flat_insts.extend(block.insts)
    if not flat_insts:
        return ir
    result = _partial_redundancy_elimination({"instructions": flat_insts})
    new_insts = result["instructions"]
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