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
            "constant_expression_query",
            [
                {"op": "add", "args": [3, 7], "out": "x"},
                {"op": "mul", "args": ["x", 2], "out": "y"},
                {"op": "add", "args": [10, 20], "out": "z"},
                {"op": "mul", "args": ["z", 3], "out": "w"},
                {"op": "add", "args": ["y", "w"], "out": "t"},
            ],
            "Simplify constant expressions in query predicates to reduce execution cost.",
        ),

        _task(
            "projection_simplification",
            [
                {"op": "const", "args": [5], "out": "a"},
                {"op": "id", "args": ["a"], "out": "b"},
                {"op": "id", "args": ["b"], "out": "c"},
                {"op": "add", "args": ["c", 3], "out": "d"},
                {"op": "add", "args": ["a", 3], "out": "e"},
                {"op": "mul", "args": ["d", 2], "out": "f"},
            ],
            "Eliminate redundant column mappings and simplify projections.",
        ),

        _task(
            "subquery_reuse_local",
            [
                {"op": "add", "args": [1, 2], "out": "x"},
                {"op": "add", "args": [1, 2], "out": "y"},
                {"op": "mul", "args": ["x", 4], "out": "z"},
                {"op": "mul", "args": ["y", 4], "out": "u"},
                {"op": "add", "args": ["z", "u"], "out": "v"},
            ],
            "Detect and reuse identical sub-expressions within a query block.",
        ),

        _task(
            "unused_projection_elimination",
            [
                {"op": "const", "args": [9], "out": "a"},
                {"op": "const", "args": [1], "out": "b"},
                {"op": "add", "args": ["a", "b"], "out": "c"},
                {"op": "mul", "args": ["c", 2], "out": "d"},
                {"op": "add", "args": [4, 5], "out": "junk1"},
                {"op": "mul", "args": [7, 8], "out": "junk2"},
                {"op": "add", "args": ["d", 0], "out": "res"},
            ],
            "Remove unused computations and redundant projections.",
        ),

        _task(
            "complex_query_optimization",
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
            "Mixed workload involving folding, reuse, and pruning optimizations.",
        ),

        _task(
            "already_optimized_query",
            [
                {"op": "const", "args": [1], "out": "x"},
                {"op": "add", "args": ["x", 2], "out": "y"},
                {"op": "mul", "args": ["y", 3], "out": "z"},
            ],
            "Query is already efficient; optimizer should terminate early.",
        ),

        _task(
            "table_write_read_cleanup",
            [
                {"op": "const", "args": [1024], "out": "ptr"},
                {"op": "const", "args": [42], "out": "val"},
                {"op": "store", "args": ["val", "ptr"], "out": None},
                {"op": "load", "args": ["ptr"], "out": "x"},
                {"op": "add", "args": ["x", 1], "out": "y"},
                {"op": "id", "args": ["val"], "out": "final_result"},
            ],
            "Remove unnecessary reads after writes while preserving data correctness.",
        ),

        _task(
            "index_aliasing_case",
            [
                {"op": "const", "args": [2048], "out": "base"},
                {"op": "add", "args": ["base", 4], "out": "addr"},
                {"op": "const", "args": [99], "out": "data"},
                {"op": "store", "args": ["data", "addr"], "out": None},
                {"op": "add", "args": [5, 5], "out": "junk"},
                {"op": "load", "args": ["addr"], "out": "result"},
            ],
            "Ensure correctness under pointer/index aliasing while pruning unused ops.",
        ),

        _task(
            "cross_expression_reuse",
            [
                {"op": "add", "args": ["a", "b"], "out": "x"},
                {"op": "mul", "args": ["c", 10], "out": "y"},
                {"op": "add", "args": ["a", "b"], "out": "z"},
                {"op": "mul", "args": ["c", 10], "out": "w"},
                {"op": "add", "args": ["z", "w"], "out": "res"},
            ],
            "Identify repeated expressions across the query and reuse them.",
        )
    ]