def grade(initial_cost, final_cost):
    if initial_cost == 0:
        return 1.0
    score = (initial_cost - final_cost) / initial_cost
    return max(0.0, min(1.0, score))

# --- OpenEnv Grader Wrappers ---

def _extract_final_count(data) -> int:
    """Helper to safely extract the instruction count from OpenEnv's payload."""
    if isinstance(data, list): 
        # If OpenEnv passes a full trajectory history
        return data[-1].get("num_instructions", 0)
    # If OpenEnv passes the final state dictionary
    return data.get("num_instructions", 0)

def grade_easy(data) -> float:
    final_count = _extract_final_count(data)
    return grade(2, final_count)  # Easy starts with 2 instructions

def grade_medium(data) -> float:
    final_count = _extract_final_count(data)
    return grade(3, final_count)  # Medium starts with 3 instructions

def grade_hard(data) -> float:
    final_count = _extract_final_count(data)
    return grade(6, final_count)  # Hard starts with 6 instructions