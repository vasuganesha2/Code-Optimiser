from typing import Any, Dict, List, Optional, Set, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Primitive Helper Functions (consumed by passes.py, env.py, and internally)
# ──────────────────────────────────────────────────────────────────────────────

_MATH_OPS: Set[str] = {
    "add", "sub", "mul", "div", "mod", "neg",
    "and", "or", "xor", "not", "shl", "shr",
    "lt", "le", "gt", "ge", "eq", "ne",
}
_TERMINATOR_OPS: Set[str] = {"ret", "return", "stop", "jmp", "br", "cbr"}
_SIDE_EFFECT_OPS: Set[str] = {"store", "print", "call"} | _TERMINATOR_OPS


def is_int(value: Any) -> bool:
    """Returns True if value is a Python int (not bool)."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_math(op: str) -> bool:
    """Returns True if op is a pure arithmetic / logical operation."""
    return op in _MATH_OPS


def is_terminator(op: str) -> bool:
    """Returns True if op ends a basic block."""
    return op in _TERMINATOR_OPS


def is_pure(op: str) -> bool:
    """Returns True if the instruction has no observable side effects."""
    return op not in _SIDE_EFFECT_OPS


def is_constant(op: str, args: List[Any]) -> bool:
    """Returns True iff this instruction is a literal constant load."""
    return op == "const" and len(args) == 1 and isinstance(args[0], (int, float))


def can_fold(op: str, args: List[Any]) -> bool:
    """Returns True if the operation can be evaluated at compile time."""
    return is_math(op) and all(isinstance(a, (int, float)) for a in args)


def evaluate(op: str, args: List[Any]) -> Optional[Any]:
    """
    Evaluate a constant-foldable expression.
    Returns None on failure (e.g. division by zero).
    """
    try:
        a = args[0] if args else None
        b = args[1] if len(args) > 1 else None
        if op == "add":  return a + b
        if op == "sub":  return a - b
        if op == "mul":  return a * b
        if op == "div":
            if b == 0:   return None
            return a / b
        if op == "mod":
            if b == 0:   return None
            return a % b
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


_COMMUTATIVE_OPS: Set[str] = {"add", "mul", "and", "or", "xor", "eq", "ne"}


def normalize(op: str, args: List[Any]) -> Tuple:
    """
    Return a canonical key for an expression so that equivalent
    computations (including commutative reorderings) map to the same key.
    All args are converted to strings for hashability.
    """
    str_args = [str(a) for a in args]
    if op in _COMMUTATIVE_OPS:
        str_args = sorted(str_args)
    return (op, *str_args)

# ──────────────────────────────────────────────────────────────────────────────
# Data-Flow Analysis Helpers (Section 9.2 Framework)
# ──────────────────────────────────────────────────────────────────────────────

def _build_graph(instructions: List[Dict[str, Any]]) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Builds an instruction-level CFG returning successors and predecessors."""
    n = len(instructions)
    succs = {i: [] for i in range(n)}
    preds = {i: [] for i in range(n)}

    for i, inst in enumerate(instructions):
        op = inst["op"]
        if op in {"ret", "return", "stop"}: continue
        args = inst.get("args", [])
        if op == "jmp" and args and isinstance(args[0], int):
            tgt = args[0]
            if 0 <= tgt < n:
                succs[i].append(tgt); preds[tgt].append(i)
            continue
        if op in {"br", "cbr"} and len(args) >= 2:
            for tgt in args[:2]:
                if isinstance(tgt, int) and 0 <= tgt < n:
                    succs[i].append(tgt); preds[tgt].append(i)
            continue
        if i + 1 < n:
            succs[i].append(i + 1); preds[i + 1].append(i)
    return succs, preds


def _reaching_definitions(instructions: List[Dict[str, Any]], succs: Dict[int, List[int]], preds: Dict[int, List[int]]) -> List[Set[Tuple[str, int]]]:
    n = len(instructions)
    IN, OUT = [set() for _ in range(n)], [set() for _ in range(n)]
    GEN, KILL = [set() for _ in range(n)], [set() for _ in range(n)]
    
    def_locs = {}
    for i, inst in enumerate(instructions):
        out_var = inst.get("out")
        if out_var:
            if out_var not in def_locs: def_locs[out_var] = set()
            def_locs[out_var].add(i)

    for i, inst in enumerate(instructions):
        out_var = inst.get("out")
        if out_var:
            GEN[i].add((out_var, i))
            KILL[i] = {(out_var, loc) for loc in def_locs[out_var] if loc != i}

    changed = True
    while changed:
        changed = False
        for i in range(n):
            new_in = set()
            for p in preds[i]: new_in.update(OUT[p])
            IN[i] = new_in
            new_out = GEN[i].union(IN[i] - KILL[i])
            if OUT[i] != new_out: OUT[i] = new_out; changed = True
    return IN


# ──────────────────────────────────────────────────────────────────────────────
# Optimization Passes
# ──────────────────────────────────────────────────────────────────────────────

def loop_invariant_code_motion(program: List[Any]) -> List[Any]:
    """
    Identifies loops via back-edges and hoists invariant instructions.
    (Kept relatively similar as loop analysis heavily relies on dominator trees, 
    but updated to respect basic invariants).
    """
    insts = [i.model_dump() if hasattr(i, 'model_dump') else i for i in program]
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
        loop_indices = range(header, tail + 1)
        defs_in_loop = {new_insts[i].get("out") for i in loop_indices if new_insts[i].get("out")}

        invariants_to_hoist = []
        for i in loop_indices:
            inst = new_insts[i]
            op, args, out = inst.get("op"), inst.get("args", []), inst.get("out")

            if (is_pure(op) and is_math(op) and out and
                all(a not in defs_in_loop for a in args if isinstance(a, str))):
                invariants_to_hoist.append(i)

        for i in reversed(invariants_to_hoist):
            moving_inst = new_insts.pop(i)
            new_insts.insert(header, moving_inst)
            header += 1 

    from env.models import Instruction
    return [Instruction(**i) for i in new_insts]


def copy_prop(program: Dict[str, Any]) -> Dict[str, Any]:
    """Global Copy Propagation using Reaching Definitions."""
    insts = program.get("instructions", [])
    if not insts: return program

    succs, preds = _build_graph(insts)
    IN_sets = _reaching_definitions(insts, succs, preds)

    new_insts = []
    for i, inst in enumerate(insts):
        args = inst.get("args", [])
        new_args = list(args)

        for arg_idx, arg in enumerate(args):
            if isinstance(arg, str):
                # Find all definitions of 'arg' that reach this point
                reaching_defs = [def_loc for def_var, def_loc in IN_sets[i] if def_var == arg]
                
                # If all reaching definitions are 'id' operations copying the SAME variable, propagate it
                if reaching_defs:
                    propagate_var = None
                    can_propagate = True
                    for loc in reaching_defs:
                        def_inst = insts[loc]
                        if def_inst["op"] == "id" and len(def_inst["args"]) == 1 and isinstance(def_inst["args"][0], str):
                            if propagate_var is None:
                                propagate_var = def_inst["args"][0]
                            elif propagate_var != def_inst["args"][0]:
                                can_propagate = False # Definitions disagree
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
    if not insts: return program

    succs, preds = _build_graph(insts)
    IN_sets = _reaching_definitions(insts, succs, preds)

    new_insts = []
    for i, inst in enumerate(insts):
        op, args, out = inst["op"], inst.get("args", []), inst.get("out")
        new_args = list(args)

        for arg_idx, arg in enumerate(args):
            if isinstance(arg, str):
                reaching_defs = [def_loc for def_var, def_loc in IN_sets[i] if def_var == arg]
                
                if reaching_defs:
                    propagate_val = None
                    can_propagate = True
                    for loc in reaching_defs:
                        def_inst = insts[loc]
                        if def_inst["op"] == "const" and len(def_inst["args"]) == 1 and isinstance(def_inst["args"][0], (int, float)):
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

        # Fold if operation is computable
        if can_fold(op, new_args) and out:
            result = evaluate(op, new_args)
            if result is not None:
                new_insts.append({"op": "const", "args": [result], "out": out})
                continue

        new_insts.append({**inst, "args": new_args})

    program["instructions"] = new_insts
    return program


def dead_code_elim(program: Dict[str, Any]) -> Dict[str, Any]:
    """Global Dead Code Elimination using Liveness Analysis Iteration."""
    insts = program.get("instructions", [])
    if not insts: return program

    n = len(insts)
    succs, preds = _build_graph(insts)
    
    IN: List[Set[str]] = [set() for _ in range(n)]
    OUT: List[Set[str]] = [set() for _ in range(n)]

    # Iterative Liveness Analysis (Backward)
    changed = True
    while changed:
        changed = False
        for i in reversed(range(n)):
            inst = insts[i]
            
            # OUT[i] = Union of IN[s] for s in succs[i]
            new_out = set()
            for s in succs[i]:
                new_out.update(IN[s])
            OUT[i] = new_out

            # IN[i] = USE[i] U (OUT[i] - DEF[i])
            new_in = set(OUT[i])
            out_var = inst.get("out")
            if out_var:
                new_in.discard(out_var)
                
            # Add USEs
            for arg in inst.get("args", []):
                if isinstance(arg, str):
                    new_in.add(arg)
                    
            if IN[i] != new_in:
                IN[i] = new_in
                changed = True

    # Elimination Phase
    new_insts = []
    for i, inst in enumerate(insts):
        op, out = inst["op"], inst.get("out")
        
        # Keep if it is a terminator, has side effects, or its output is live globally
        if is_terminator(op) or not is_pure(op) or (out and out in OUT[i]):
            new_insts.append(inst)

    program["instructions"] = new_insts
    return program




def _available_expressions(instructions: List[Dict[str, Any]], succs: Dict[int, List[int]], preds: Dict[int, List[int]]) -> List[Dict[Any, str]]:
    """
    Computes Global Available Expressions (Intersection-based).
    Returns IN sets for each instruction: Map of expression_key -> variable_holding_result.
    """
    n = len(instructions)
    # 1. Identify all candidate expressions and where they are killed
    # expression_key -> Set of indices where it is computed
    all_exprs = {}
    for i, inst in enumerate(instructions):
        if is_math(inst["op"]) and is_pure(inst["op"]):
            key = normalize(inst["op"], inst.get("args", []))
            if key not in all_exprs: all_exprs[key] = set()
            all_exprs[key].add(i)

    # GEN[i]: expression generated by instruction i
    # KILL[i]: all expressions that use the variable defined in instruction i
    GEN = [None] * n
    KILL = [set() for _ in range(n)]
    for i, inst in enumerate(instructions):
        if is_math(inst["op"]) and is_pure(inst["op"]):
            GEN[i] = normalize(inst["op"], inst.get("args", []))
        
        out_var = inst.get("out")
        if out_var:
            for expr_key in all_exprs:
                # If an expression uses this out_var as an argument, it is killed
                if out_var in expr_key[1]:
                    KILL[i].add(expr_key)

    # 2. Iterative Solver (Intersection)
    # Initial OUT: OUT[entry] is empty, all others are the set of all expressions
    universe = set(all_exprs.keys())
    OUT = [set(universe) for _ in range(n)]
    if n > 0: OUT[0] = set() 

    changed = True
    while changed:
        changed = False
        for i in range(n):
            # IN[i] = Intersect OUT of predecessors
            if not preds[i]:
                new_in = set()
            else:
                new_in = set.intersection(*(OUT[p] for p in preds[i]))
            
            # OUT[i] = GEN[i] U (IN[i] - KILL[i])
            # If GEN[i] is defined, add it. If KILL[i] intersects, remove it.
            temp_out = new_in - KILL[i]
            if GEN[i]:
                temp_out.add(GEN[i])
            
            if OUT[i] != temp_out:
                OUT[i] = temp_out
                changed = True

    # 3. Map expressions to a reaching variable
    # This is a simplification: we find the nearest variable holding the expr.
    results = []
    for i in range(n):
        expr_to_var = {}
        for expr in OUT[i]:
            # For each available expr, find which variable currently holds it
            # We look at the reaching definitions for the result of that expr
            for loc in all_exprs[expr]:
                if loc < i: # Must be defined before this point
                    expr_to_var[expr] = instructions[loc]["out"]
        results.append(expr_to_var)
    return results


def local_cse(program: Dict[str, Any]) -> Dict[str, Any]:
    """Now updated to Global Common Subexpression Elimination."""
    insts = program.get("instructions", [])
    if not insts: return program

    succs, preds = _build_graph(insts)
    available_in = _available_expressions(insts, succs, preds)

    new_insts = []
    for i, inst in enumerate(insts):
        op, args, out = inst["op"], inst.get("args", []), inst.get("out")
        
        if is_math(op) and is_pure(op):
            expr_key = normalize(op, args)
            # If expression is available in the IN set, replace with a copy (id)
            if expr_key in available_in[i]:
                prev_var = available_in[i][expr_key]
                new_insts.append({"op": "id", "args": [prev_var], "out": out})
                continue
        
        new_insts.append(inst)

    program["instructions"] = new_insts
    return program


def global_constant_propagation(program: Dict[str, Any]) -> Dict[str, Any]:
    insts = program.get("instructions", [])
    if not insts: return program
    n = len(insts)
    succs, preds = _build_graph(insts)

    # V: Map of variable -> Lattice Value (None = UNDEF, "NAC" = Not a Constant)
    # IN/OUT: List of maps for each instruction
    IN = [{} for _ in range(n)]
    OUT = [{} for _ in range(n)]

    def meet(v1, v2):
        if v1 is None: return v2
        if v2 is None: return v1
        if v1 == "NAC" or v2 == "NAC": return "NAC"
        if v1 != v2: return "NAC"
        return v1

    changed = True
    while changed:
        changed = False
        for i in range(n):
            # 1. Meet operator (Intersection of values)
            new_in = {}
            # Get union of all variables across all predecessors
            all_vars = set().union(*(OUT[p].keys() for p in preds[i]))
            for v in all_vars:
                vals = [OUT[p].get(v) for p in preds[i]]
                res = vals[0] if vals else None
                for other in vals[1:]:
                    res = meet(res, other)
                new_in[v] = res
            
            IN[i] = new_in
            
            # 2. Transfer Function (Section 9.4.3 logic)
            new_out = dict(new_in)
            inst = insts[i]
            out_var = inst.get("out")
            if out_var:
                op, args = inst["op"], inst.get("args", [])
                if op == "const":
                    new_out[out_var] = args[0]
                elif is_math(op):
                    # RHS logic: y + z
                    arg_vals = []
                    for a in args:
                        val = new_in.get(a) if isinstance(a, str) else a
                        arg_vals.append(val)
                    
                    if any(v == "NAC" for v in arg_vals):
                        new_out[out_var] = "NAC"
                    elif all(isinstance(v, (int, float)) for v in arg_vals):
                        new_out[out_var] = evaluate(op, arg_vals)
                    else:
                        new_out[out_var] = None # Still UNDEF
                else:
                    new_out[out_var] = "NAC"

            if OUT[i] != new_out:
                OUT[i] = new_out
                changed = True

    # Apply findings
    new_insts = []
    for i, inst in enumerate(insts):
        new_args = []
        for a in inst.get("args", []):
            if isinstance(a, str) and isinstance(IN[i].get(a), (int, float)):
                new_args.append(IN[i][a])
            else:
                new_args.append(a)
        new_insts.append({**inst, "args": new_args})
    
    program["instructions"] = new_insts
    return program


def partial_redundancy_elimination(program: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full Implementation of Algorithm 9.36: Lazy Code Motion.
    This replaces the heuristic version with a mathematically optimal 4-pass analysis.
    """
    insts = program.get("instructions", [])
    if not insts: return program
    n = len(insts)
    succs, preds = _build_graph(insts)
    
    # --- STEP 1: Identify Universal Set (U), e_use, and e_kill ---
    U: Set[Tuple] = set()
    e_use = [set() for _ in range(n)]
    e_kill = [set() for _ in range(n)]
    
    for i, inst in enumerate(insts):
        op, args, out = inst["op"], inst.get("args", []), inst.get("out")
        if is_math(op) and is_pure(op):
            expr = normalize(op, args)
            U.add(expr)
            # Use logic: expression is used if operands aren't redefined earlier in the block
            # (In instruction-level, we treat each instruction as its own tiny block)
            e_use[i].add(expr)
        
        if out:
            for expr in U:
                if out in expr[1]: # If out_var is an operand in the expression
                    e_kill[i].add(expr)

    def solve(forward: bool, intersect: bool, transfer_func, initial_set):
        # Data-flow solver helper
        nodes = list(range(n))
        if not forward: nodes.reverse()
        
        IN = [set(initial_set) for _ in range(n)]
        OUT = [set(initial_set) for _ in range(n)]
        
        # Boundary
        if forward: IN[0] = set()
        else: OUT[n-1] = set()

        changed = True
        while changed:
            changed = False
            for i in nodes:
                neighbors = preds[i] if forward else succs[i]
                if neighbors:
                    combined = set(OUT[neighbors[0]] if forward else IN[neighbors[0]])
                    for nb in neighbors[1:]:
                        if intersect: combined &= (OUT[nb] if forward else IN[nb])
                        else: combined |= (OUT[nb] if forward else IN[nb])
                else:
                    combined = set()

                if forward: IN[i] = combined
                else: OUT[i] = combined

                new_val = transfer_func(i, IN[i] if forward else OUT[i])
                target = OUT if forward else IN
                if target[i] != new_val:
                    target[i] = new_val
                    changed = True
        return IN, OUT

    # --- STEP 2: Anticipated Expressions (Backward, Intersect) ---
    antic_in, _ = solve(False, True, lambda i, out_i: e_use[i] | (out_i - e_kill[i]), U)

    # --- STEP 3: Available Expressions (Forward, Intersect) ---
    # Transfer: (antic_in[i] | in_i) - e_kill[i]
    avail_in, _ = solve(True, True, lambda i, in_i: (antic_in[i] | in_i) - e_kill[i], U)

    # --- STEP 4: Earliest ---
    earliest = [antic_in[i] - avail_in[i] for i in range(n)]

    # --- STEP 5: Postponable Expressions (Forward, Intersect) ---
    # Transfer: (earliest[i] | in_i) - e_use[i]
    post_in, _ = solve(True, True, lambda i, in_i: (earliest[i] | in_i) - e_use[i], U)

    # --- STEP 6: Latest ---
    latest = [set() for _ in range(n)]
    for i in range(n):
        ear_post = earliest[i] | post_in[i]
        # Successor Intersection
        succ_comp = set(U)
        if succs[i]:
            for s in succs[i]: succ_comp &= (earliest[s] | post_in[s])
        else: succ_comp = set()
        
        latest[i] = ear_post & (e_use[i] | (U - succ_comp))

    # --- STEP 7: Used Expressions (Backward, Union) ---
    # Transfer: (e_use[i] | out_i) - latest[i]
    _, used_out = solve(False, False, lambda i, out_i: (e_use[i] | out_i) - latest[i], set())

    # --- STEP 8: Transformation ---
    # 8a: Place temp assignments at Latest & Used points
    # 8b: Replace original computations with temp access
    new_insts = []
    expr_to_temp = {expr: f"t_pre_{idx}" for idx, expr in enumerate(U)}
    
    for i, inst in enumerate(insts):
        # Insert calculations for expressions that transition here
        for expr in (latest[i] & used_out[i]):
            new_insts.append({"op": expr[0], "args": list(expr[1]), "out": expr_to_temp[expr]})
        
        # Rewrite existing math ops if they are now redundant
        op, args, out = inst["op"], inst.get("args", []), inst.get("out")
        if is_math(op) and is_pure(op):
            expr = normalize(op, args)
            if expr in (used_out[i] | latest[i]):
                new_insts.append({"op": "id", "args": [expr_to_temp[expr]], "out": out})
                continue
        
        new_insts.append(inst)

    program["instructions"] = new_insts
    return program




def noop(program: Dict[str, Any]) -> Dict[str, Any]:
    return program