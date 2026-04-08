from pydantic import BaseModel
from typing import List, Union

class Instruction(BaseModel):
    op: str
    args: List[Union[int, str]]
    out: str

class Observation(BaseModel):
    num_instructions: int
    steps_left: int
    last_action: str
    program: List[Instruction]


class Action(BaseModel):
    pass_name: str


class Reward(BaseModel):
    value: float