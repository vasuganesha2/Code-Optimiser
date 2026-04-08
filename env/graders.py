def grade(initial_cost, final_cost):
    if initial_cost == 0:
        return 1.0
    score = (initial_cost - final_cost) / initial_cost
    return max(0.0, min(1.0, score))

