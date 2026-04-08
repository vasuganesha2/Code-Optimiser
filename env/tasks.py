def get_tasks():
    return [
        # ── EASY ──────────────────────────────────────────────────────────────
        # Optimal: const_fold → stop (1 instruction removed)
        {
            "name": "easy",
            "program": {
                "instructions": [
                    {"op": "add",  "args": [3, 7],    "out": "x"},   # foldable → 10
                    {"op": "mul",  "args": ["x", 2],  "out": "y"},
                ]
            }
        },

        # ── MEDIUM ────────────────────────────────────────────────────────────
        # Optimal: const_fold → dead_code_elim → stop (2 instructions removed)
        # const_fold turns 'w' into a const; 'w' is never used → becomes dead
        {
            "name": "medium",
            "program": {
                "instructions": [
                    {"op": "add",  "args": [1, 2],    "out": "w"},   # foldable → 3, but unused
                    {"op": "add",  "args": [4, 5],    "out": "x"},   # foldable → 9
                    {"op": "mul",  "args": ["x", 3],  "out": "y"},
                ]
            }
        },

        # ── HARD ──────────────────────────────────────────────────────────────
        # Optimal: const_fold × 2 → dead_code_elim × 2 → stop (4 removed)
        # Tests whether agent sequences multiple passes without wasting steps
        {
            "name": "hard",
            "program": {
                "instructions": [
                    {"op": "add",  "args": [1, 1],    "out": "a"},   # → 2
                    {"op": "mul",  "args": [3, 3],    "out": "b"},   # → 9, dead
                    {"op": "add",  "args": ["a", 0],  "out": "c"},   # not foldable (var)
                    {"op": "mul",  "args": [2, 4],    "out": "d"},   # → 8, dead
                    {"op": "add",  "args": ["c", 1],  "out": "e"},   # dead
                    {"op": "mul",  "args": ["a", 3],  "out": "y"},
                ]
            }
        },
    ]