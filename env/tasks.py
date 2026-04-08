# env/tasks.py

from env.graders import grade

# Small epsilon to keep scores strictly within (0,1)
_EPSILON = 1e-9

def get_tasks():
    """
    Returns 3 tasks, each with a grader.
    Each grader captures the initial instruction count so
    the score is computed correctly against the starting program.
    """
    return [
        {
            "name": "easy",
            "program": {
                "instructions": [
                    {"op": "add", "args": [3, 7], "out": "x"},
                    {"op": "mul", "args": ["x", 2], "out": "y"},
                ]
            },
            "grader": grade(2),  # initial_count = number of instructions
        },
        {
            "name": "medium",
            "program": {
                "instructions": [
                    {"op": "add", "args": [1, 2], "out": "w"},
                    {"op": "add", "args": [4, 5], "out": "x"},
                    {"op": "mul", "args": ["x", 3], "out": "y"},
                ]
            },
            "grader": grade(3),
        },
        {
            "name": "hard",
            "program": {
                "instructions": [
                    {"op": "add", "args": [1, 1], "out": "a"},
                    {"op": "mul", "args": [3, 3], "out": "b"},
                    {"op": "add", "args": ["a", 0], "out": "c"},
                    {"op": "mul", "args": [2, 4], "out": "d"},
                    {"op": "add", "args": ["c", 1], "out": "e"},
                    {"op": "mul", "args": ["a", 3], "out": "y"},
                ]
            },
            "grader": grade(6),
        },
    ]