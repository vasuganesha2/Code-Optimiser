# env/graders.py

_EPSILON = 1e-9  # ensures score strictly in (0,1)

def grade(initial_count: int):
    """
    Returns a grader closure for a task.
    Scores are clamped within (0,1) to satisfy HF validation.
    """
    def grader(observation, reward=None, done=None):
        program_data = observation.get("program", [])
        if isinstance(program_data, list):
            final_count = len(program_data)
        elif isinstance(program_data, dict):
            final_count = len(program_data.get("instructions", []))
        else:
            final_count = initial_count

        raw_score = (initial_count - final_count) / initial_count
        # Clamp strictly within (0,1)
        return float(max(_EPSILON, min(1.0 - _EPSILON, raw_score)))

    grader.__name__ = "grader"
    return grader