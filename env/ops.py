from typing import List, Optional

COMMUTATIVE_OPS = {"add", "mul"}
MATH_OPS = {"add", "mul", "sub", "neg"}
TERMINATORS = {"ret", "return", "stop", "jmp", "br", "cbr"}
PURE_OPS = {"const", "id", "add", "mul", "sub", "neg"}


# ── basic checks ───────────────────────────────────────────────────────────

def is_int(x) -> bool:
    """Strict int check (excludes bool)."""
    return isinstance(x, int) and not isinstance(x, bool)


def is_commutative(op: str) -> bool:
    return op in COMMUTATIVE_OPS


def is_math(op: str) -> bool:
    return op in MATH_OPS


def is_terminator(op: str) -> bool:
    return op in TERMINATORS


def is_pure(op: str) -> bool:
    """Safe for code motion (no side effects)."""
    return op in PURE_OPS


# ── constant folding ───────────────────────────────────────────────────────

def is_constant(op: str, args: List) -> bool:
    return op == "const" and len(args) == 1 and is_int(args[0])


def can_fold(op: str, args: List) -> bool:
    return is_math(op) and all(is_int(a) for a in args)


def evaluate(op: str, args: List[int]) -> Optional[int]:
    try:
        if op == "add":
            return args[0] + args[1]
        if op == "mul":
            return args[0] * args[1]
        if op == "sub":
            return args[0] - args[1]
        if op == "neg":
            return -args[0]
    except Exception:
        return None
    return None


# ── normalization (for CSE) ────────────────────────────────────────────────

def normalize(op: str, args: List):
    """Canonical form of expression (for CSE)."""
    if is_commutative(op):
        return op, tuple(sorted(args, key=lambda x: str(x)))
    return op, tuple(args)


# ── cost model (VERY IMPORTANT for RL) ─────────────────────────────────────

OP_COST = {
    "const": 0,
    "id": 0,
    "add": 1,
    "sub": 1,
    "neg": 1,
    "mul": 2,
}


def op_cost(op: str) -> int:
    return OP_COST.get(op, 1)