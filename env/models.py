from typing import List, Union, Optional, Dict, Any
from pydantic import BaseModel, field_validator


class Instruction(BaseModel):
    op: str
    args: List[Union[int, float, str]] = []
    out: Optional[str] = None

    @field_validator("op")
    @classmethod
    def op_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("op must be a non-empty string")
        return v.lower().strip()

    def is_terminator(self) -> bool:
        """Returns True if this instruction ends a basic block."""
        return self.op in {"ret", "return", "stop", "jmp", "br", "cbr"}

    def uses(self) -> List[str]:
        """Variable names read by this instruction."""
        return [a for a in self.args if isinstance(a, str)]

    def defines(self) -> Optional[str]:
        """Variable name written by this instruction (if any)."""
        return self.out

    def __repr__(self) -> str:
        lhs = f"{self.out} = " if self.out else ""
        args_str = ", ".join(str(a) for a in self.args)
        return f"{lhs}{self.op}({args_str})"


class Observation(BaseModel):
    # ── core program state ──────────────────────────────────────────────────
    num_instructions: int
    steps_left: int
    last_action: str
    program: List[Instruction]

    # ── graph structure ─────────────────────────────────────────────────────
    cfg: Dict[str, Any]           # {"entry": int|None, "edges": {i:[j,...]}, "exit": [int]}

    # ── derived features (computed by env, consumed by agent) ───────────────
    op_counts: Dict[str, int]     # {"const": 3, "add": 2, ...}
    live_vars: List[str]          # variables live at program entry (backward analysis)
    defined_vars: List[str]       # variables assigned at least once in the program


class Action(BaseModel):
    pass_name: str

    @field_validator("pass_name")
    @classmethod
    def must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("pass_name must be a non-empty string")
        return v.strip()


class Reward(BaseModel):
    value: float

    @field_validator("value")
    @classmethod
    def must_be_finite(cls, v: float) -> float:
        import math
        if not math.isfinite(v):
            raise ValueError("Reward value must be finite")
        return v