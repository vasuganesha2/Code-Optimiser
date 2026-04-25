from typing import Dict, Any, List
from env.ops import (
    is_int, is_math, is_terminator, is_pure, 
    is_constant, can_fold, evaluate, normalize
)



def loop_invariant_code_motion(program: List[Any]) -> List[Any]:
    """
    Identifies loops via back-edges and hoists invariant instructions 
    (instructions whose arguments are defined outside the loop).
    """
    # Convert to list of dicts if they are Pydantic models for consistency
    insts = [i.model_dump() if hasattr(i, 'model_dump') else i for i in program]
    
    if not insts:
        return program

    # 1. Identify Back-edges (a simple way to find loops in linear IR)
    # A back-edge is a jump/branch from index 'j' to index 'i' where i <= j.
    loops = []
    for idx, inst in enumerate(insts):
        op = inst.get("op")
        args = inst.get("args", [])
        if op in {"jmp", "br", "cbr"}:
            for arg in args:
                if isinstance(arg, int) and arg <= idx:
                    loops.append((arg, idx)) # (header_idx, back_edge_idx)

    if not loops:
        return program

    new_insts = list(insts)
    
    for header, tail in loops:
        loop_indices = range(header, tail + 1)
        
        # 2. Find variables defined inside the loop
        defs_in_loop = set()
        for i in loop_indices:
            out = new_insts[i].get("out")
            if out:
                defs_in_loop.add(out)

        # 3. Identify Invariants
        invariants_to_hoist = []
        for i in loop_indices:
            inst = new_insts[i]
            op = inst.get("op")
            args = inst.get("args", [])
            out = inst.get("out")

            # Condition: Pure math op AND all string args are NOT defined in this loop
            is_invariant = (
                is_pure(op) and is_math(op) and out and
                all(a not in defs_in_loop for a in args if isinstance(a, str))
            )

            if is_invariant:
                invariants_to_hoist.append(i)

        # 4. Perform Hoisting (Move to right before the header)
        # We process in reverse to maintain correct indexing while popping
        for i in reversed(invariants_to_hoist):
            moving_inst = new_insts.pop(i)
            new_insts.insert(header, moving_inst)
            # Adjust header because we just inserted before it
            header += 1 

    # Return as original types (convert back to Instruction if needed)
    from env.models import Instruction
    return [Instruction(**i) for i in new_insts]

def copy_prop(program: Dict[str, Any]) -> Dict[str, Any]:
    """Substitute variables that are direct copies of other variables."""
    instructions = program.get("instructions", [])
    copy_map = {}
    new_insts = []

    for inst in instructions:
        # Use copy_map to replace arguments if they are variables
        new_args = [copy_map.get(a, a) for a in inst.get("args", [])]
        new_inst = {**inst, "args": new_args}
        out = inst.get("out")

        # Check for 'id' instruction: x = id y
        if inst["op"] == "id" and len(new_args) == 1 and isinstance(new_args[0], str) and out:
            copy_map[out] = new_args[0]
        else:
            # Invalidation: if this instruction writes to a variable in copy_map, clear it
            if out in copy_map:
                del copy_map[out]
            copy_map = {k: v for k, v in copy_map.items() if v != out}

        new_insts.append(new_inst)

    program["instructions"] = new_insts
    return program


def const_fold(program: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constant folding using helpers from ops.py.
    """
    instructions = program.get("instructions", [])
    const_map = {}
    
    # Phase 1: Build initial map from existing 'const' ops
    for inst in instructions:
        op, args, out = inst["op"], inst.get("args", []), inst.get("out")
        if is_constant(op, args) and out:
            const_map[out] = args[0]

    new_insts = []
    for inst in instructions:
        op, out = inst["op"], inst.get("out")
        
        # Substitute known constants into args
        new_args = [
            const_map[a] if (isinstance(a, str) and a in const_map) else a
            for a in inst.get("args", [])
        ]

        # Fold if the operation is now computable
        if can_fold(op, new_args) and out:
            result = evaluate(op, new_args)
            if result is not None:
                new_insts.append({"op": "const", "args": [result], "out": out})
                const_map[out] = result
                continue

        # Invalidation and Update
        if out:
            # If we are overwriting a variable with a non-constant, remove it from map
            if not is_constant(op, new_args):
                const_map.pop(out, None)
            else:
                const_map[out] = new_args[0]

        new_insts.append({**inst, "args": new_args})

    program["instructions"] = new_insts
    return program


def dead_code_elim(program: Dict[str, Any]) -> Dict[str, Any]:
    """Eliminates instructions whose outputs are never used."""
    live = set()
    new_insts = []
    instructions = program.get("instructions", [])

    if not instructions:
        return program

    # Traverse backward to determine liveness
    for inst in reversed(instructions):
        op, args, out = inst["op"], inst.get("args", []), inst.get("out")

        # Terminators always stay; their args become live
        if is_terminator(op):
            new_insts.append(inst)
            for arg in args:
                if isinstance(arg, str): live.add(arg)
            continue

        # Keep instruction if it has side effects (not pure) or its output is live
        if not is_pure(op) or (out and out in live):
            new_insts.append(inst)
            if out in live:
                live.remove(out)
            for arg in args:
                if isinstance(arg, str): live.add(arg)

    new_insts.reverse()
    program["instructions"] = new_insts
    return program


def local_cse(program: Dict[str, Any]) -> Dict[str, Any]:
    """Common Subexpression Elimination using normalization."""
    instructions = program.get("instructions", [])
    new_insts = []
    # Map: (op, (args)) -> variable_name
    expr_table = {}

    for inst in instructions:
        op, args, out = inst["op"], inst.get("args", []), inst.get("out")

        # 1. Normalize and check for existing computation
        expr_key = normalize(op, args)
        
        if is_math(op) and expr_key in expr_table and out:
            previous_out = expr_table[expr_key]
            new_insts.append({"op": "id", "args": [previous_out], "out": out})
        else:
            if is_math(op) and out:
                expr_table[expr_key] = out
            new_insts.append(inst)

        # 2. Invalidation: If 'out' is modified, any expression using it or 
        # previously stored as 'out' is no longer valid.
        if out:
            keys_to_remove = [
                k for k, v in expr_table.items()
                if out in k[1] or v == out
            ]
            for k in keys_to_remove:
                del expr_table[k]

    program["instructions"] = new_insts
    return program


def noop(program: Dict[str, Any]) -> Dict[str, Any]:
    return program