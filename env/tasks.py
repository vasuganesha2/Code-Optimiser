from env.graders import grade


def _task(name, instructions, description=""):
    return {
        "name": name,
        "description": description,
        "program": {"instructions": instructions},
        "grader": grade(len(instructions)),
    }


def get_tasks():
    return [
        _task(
            "easy_fold",
            [
                {"op": "add", "args": [3, 7], "out": "x"},
                {"op": "mul", "args": ["x", 2], "out": "y"},
                {"op": "add", "args": [10, 20], "out": "z"},
                {"op": "mul", "args": ["z", 3], "out": "w"},
                {"op": "add", "args": ["y", "w"], "out": "t"},
            ],
            "Constant folding and chained simplification.",
        ),

        _task(
            "copy_chain",
            [
                {"op": "const", "args": [5], "out": "a"},
                {"op": "id", "args": ["a"], "out": "b"},
                {"op": "id", "args": ["b"], "out": "c"},
                {"op": "add", "args": ["c", 3], "out": "d"},
                {"op": "add", "args": ["a", 3], "out": "e"},
                {"op": "mul", "args": ["d", 2], "out": "f"},
            ],
            "Copy propagation should collapse the chain and expose folds.",
        ),

        _task(
            "local_cse",
            [
                {"op": "add", "args": [1, 2], "out": "x"},
                {"op": "add", "args": [1, 2], "out": "y"},
                {"op": "mul", "args": ["x", 4], "out": "z"},
                {"op": "mul", "args": ["y", 4], "out": "u"},
                {"op": "add", "args": ["z", "u"], "out": "v"},
            ],
            "Repeated expressions should be reused by local CSE.",
        ),

        _task(
            "dead_code",
            [
                {"op": "const", "args": [9], "out": "a"},
                {"op": "const", "args": [1], "out": "b"},
                {"op": "add", "args": ["a", "b"], "out": "c"},
                {"op": "mul", "args": ["c", 2], "out": "d"},
                {"op": "add", "args": [4, 5], "out": "junk1"},
                {"op": "mul", "args": [7, 8], "out": "junk2"},
                {"op": "add", "args": ["d", 0], "out": "res"},
            ],
            "Dead code elimination should remove unused junk instructions.",
        ),

        _task(
            "mixed",
            [
                {"op": "const", "args": [2], "out": "a"},
                {"op": "id", "args": ["a"], "out": "b"},
                {"op": "add", "args": ["b", 3], "out": "c"},
                {"op": "add", "args": [2, 3], "out": "d"},
                {"op": "add", "args": [2, 3], "out": "e"},
                {"op": "mul", "args": ["c", 4], "out": "f"},
                {"op": "mul", "args": ["d", 4], "out": "g"},
                {"op": "add", "args": ["f", "g"], "out": "h"},
                {"op": "add", "args": [100, 200], "out": "unused1"},
                {"op": "mul", "args": [10, 10], "out": "unused2"},
            ],
            "A mix of folding, copy propagation, CSE, and DCE.",
        ),

        _task(
            "stop_case",
            [
                {"op": "const", "args": [1], "out": "x"},
                {"op": "add", "args": ["x", 2], "out": "y"},
                {"op": "mul", "args": ["y", 3], "out": "z"},
            ],
            "Already short; baseline should terminate quickly once no pass helps.",
        ),

       _task(
            "memory_basic",
            [
                {"op": "const", "args": [1024], "out": "ptr"},
                {"op": "const", "args": [42], "out": "val"},
                {"op": "store", "args": ["val", "ptr"], "out": None}, 
                {"op": "load", "args": ["ptr"], "out": "x"},
                {"op": "add", "args": ["x", 1], "out": "y"},
                {"op": "id", "args": ["val"], "out": "final_result"} 
            ],
            "Store must be preserved. Load and add are dead and should be removed.",
        ),

        _task(
            "memory_alias",
            [
                {"op": "const", "args": [2048], "out": "base"},
                {"op": "add", "args": ["base", 4], "out": "addr"},
                {"op": "const", "args": [99], "out": "data"},
                {"op": "store", "args": ["data", "addr"], "out": None},
                {"op": "add", "args": [5, 5], "out": "junk"},
                {"op": "load", "args": ["addr"], "out": "result"},
            ],
            "Store is protected. Junk math is dead. Load is the final return.",
        ),

        _task(
            "complex_cse",
            [
                {"op": "add", "args": ["a", "b"], "out": "x"},
                {"op": "mul", "args": ["c", 10], "out": "y"},
                {"op": "add", "args": ["a", "b"], "out": "z"}, # Redundant!
                {"op": "mul", "args": ["c", 10], "out": "w"}, # Redundant!
                {"op": "add", "args": ["z", "w"], "out": "res"},
            ],
            "Uses variables to prevent const_fold from 'masking' the CSE opportunity."
        ),

        _task(
            "loop_hoist",
            [
                {"op": "const", "args": [0], "out": "i"},
                {"op": "add", "args": ["a", "b"], "out": "invariant"}, # <--- Hoist this!
                {"op": "add", "args": ["i", 1], "out": "i"},
                {"op": "br", "args": [1, 4], "out": None}, # Simple loop back-edge
                {"op": "ret", "args": ["invariant"], "out": None},
            ],
            "LICM should move the invariant addition outside the loop."
        ),

        _task(
            "lcm_branch",
            [
                {"op": "br", "args": [1, 3], "out": None}, # If/Else
                {"op": "add", "args": ["a", "b"], "out": "x"}, # Path A
                {"op": "jmp", "args": [4], "out": None},
                {"op": "add", "args": ["a", "b"], "out": "y"}, # Path B
                {"op": "ret", "args": ["x"], "out": None},
            ],
            "LCM should hoist the 'add a, b' to the header before the branch."
        ),

        _task(
            "global_dominance",
            [
                {"op": "add", "args": ["a", "b"], "out": "x"}, # Block 0 (Header)
                {"op": "jmp", "args": [2], "out": None},
                {"op": "add", "args": ["a", "b"], "out": "y"}, # Block 1 (Successor)
                {"op": "ret", "args": ["y"], "out": None},
            ],
            "Global CSE should catch the redundant add because Block 0 dominates Block 1."
        )
    ]