def const_fold(program):
    """
    Two-phase constant folding:
    1. Build a map of variable -> int from existing 'const' instructions
    2. Substitute known int values into all instruction args
    3. Fold any instruction whose args are now ALL ints
    Supports: add, mul, sub, neg
    """
    instructions = program["instructions"]

    # Phase 1: build const map from already-folded instructions
    const_map = {}
    for inst in instructions:
        if inst["op"] == "const" and len(inst["args"]) == 1 and isinstance(inst["args"][0], int):
            const_map[inst["out"]] = inst["args"][0]

    # Phase 2: substitute variable args with known int values
    substituted = []
    for inst in instructions:
        new_args = [
            const_map[a] if isinstance(a, str) and a in const_map else a
            for a in inst["args"]
        ]
        substituted.append({**inst, "args": new_args})

    # Phase 3: fold instructions whose args are now all ints
    def evaluate(op, args):
        try:
            if op == "add": return args[0] + args[1]
            if op == "mul": return args[0] * args[1]
            if op == "sub": return args[0] - args[1]
            if op == "neg": return -args[0]
        except Exception:
            return None
        return None

    new_insts = []
    for inst in substituted:
        if inst["op"] not in ("const", "noop", "stop") and all(isinstance(a, int) for a in inst["args"]):
            result = evaluate(inst["op"], inst["args"])
            if result is not None:
                new_insts.append({"op": "const", "args": [result], "out": inst["out"]})
                continue
        new_insts.append(inst)

    program["instructions"] = new_insts
    return program


def dead_code_elim(program):
    
    """
    Remove instructions whose output is never used by any later instruction
    AND is not a program output variable.

    The last instruction's output is always treated as a live program output
    so it is never incorrectly removed.
    """

    if not program["instructions"]:
        return program

    # Variables read by at least one instruction
    used = set()
    for inst in program["instructions"]:
        for arg in inst["args"]:
            if isinstance(arg, str):
                used.add(arg)

    # The final instruction's output is always live (program result)
    live_outputs = {program["instructions"][-1]["out"]}

    new_insts = []
    for inst in program["instructions"]:
        if inst["out"] in used or inst["out"] in live_outputs:
            new_insts.append(inst)
        # else: instruction is dead — drop it

    program["instructions"] = new_insts
    return program


def noop(program):
    """No-op pass — returns the program unchanged."""
    return program
