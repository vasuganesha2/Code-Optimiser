# env/tasks.py

def default_grader(observation, reward, done):
    """
    Standard instruction reduction grader. 
    Moved to top-level for validator visibility.
    """
    # Safeguard: Navigate to the actual instruction list
    program_data = observation.get("program", {})
    if isinstance(program_data, dict):
        instructions = program_data.get("instructions", [])
    else:
        instructions = program_data
        
    # Get initial count from metadata or calculate from current list
    initial = observation.get("initial_instructions", len(instructions))
    final   = observation.get("num_instructions", len(instructions))
    
    if initial == 0:
        return 0.0
        
    # Standard reduction formula
    score = (initial - final) / initial
    return max(0.0, float(score))

def get_tasks():
    """
    Returns the required 3 tasks. 
    The 'grader' key must point to the global function.
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
            "grader": default_grader,
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
            "grader": default_grader,
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
            "grader": default_grader,
        },
    ]
