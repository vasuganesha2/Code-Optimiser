# env/graders.py

def grade(initial_count, final_count):
    """
    Helper for local baseline testing.
    """
    if initial_count <= 0:
        return 1.0 if final_count == 0 else 0.0
    score = (initial_count - final_count) / initial_count
    return max(0.0, min(1.0, float(score)))
