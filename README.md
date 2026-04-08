# Compiler Optimization Environment

A reinforcement learning environment where an agent learns to optimize programs by sequencing compiler passes. Modeled after real compiler pipelines (LLVM-style IR optimization), it tests whether an agent can reason about program structure and apply the right transformations in the right order.

## Motivation

Compiler pass ordering is a genuinely hard combinatorial problem — even LLVM uses heuristics for it. This environment makes it tractable for RL agents: the state is explicit, actions are discrete, and rewards are dense. It is directly useful for evaluating agents that must reason about code structure, causal chains between transformations, and efficient use of a limited action budget.

---

## Observation Space

| Field              | Type              | Description                                  |
|--------------------|-------------------|----------------------------------------------|
| `num_instructions` | `int`             | Current number of instructions in the program |
| `steps_left`       | `int`             | Remaining steps in the episode               |
| `last_action`      | `str`             | Name of the most recently applied pass       |
| `program`          | `List[Instruction]` | Full current instruction list              |

Each `Instruction` has:
- `op` — operation name (`add`, `mul`, `const`)
- `args` — list of integer or variable-name arguments
- `out` — output variable name

---

## Action Space

| Action           | Effect |
|------------------|--------|
| `const_fold`     | Replaces instructions with all-constant args with a single `const`. Also propagates known constant variables into dependent instructions before folding. |
| `dead_code_elim` | Removes instructions whose output is never used downstream |
| `noop`           | No-op; valid but wastes a step (-0.05) |
| `stop`           | Ends the episode immediately; +1.0 bonus if nothing left to optimize |

---

## Reward Function

```
reward = (instructions_removed × 1.0)
       - (0.2  if pass did nothing)
       - (0.05 per step, always)
```

For `stop`:
- `+1.0` if no further optimization is possible (clean termination)
- `-0.2` if useful passes remain (premature stop)

This gives a dense, shaped signal that rewards both optimization quality and efficiency (not wasting steps).

---

## Tasks

### Easy — `easy`

Program: 2 instructions, one foldable constant `add` that feeds a `mul`.

```
add [3, 7]  -> x
mul [x, 2]  -> y
```

Optimal sequence: `const_fold` → `const_fold` → `dead_code_elim` → `stop`

Instructions removed: 1 (x eliminated after y is fully folded)

Difficulty: Baseline — tests whether agent can recognize a foldable instruction and propagate constants.

---

### Medium — `medium`

Program: 3 instructions — two foldable constants, one of which produces dead code after folding.

```
add [1, 2]  -> w
add [4, 5]  -> x
mul [x, 3]  -> y
```

Optimal sequence: `const_fold` → `const_fold` → `dead_code_elim` → `stop`

Instructions removed: 2 (w and x both become dead after y is folded)

Difficulty: Tests pass ordering — folding must happen before DCE reveals dead code.

---

### Hard — `hard`

Program: 6 instructions — multiple foldable constants, multiple dead vars, a non-foldable var-arg instruction acting as a distractor.

Optimal sequence: `const_fold` → `const_fold` → `dead_code_elim` → `dead_code_elim` → `stop`

Instructions removed: 3

Difficulty: Requires multi-cycle reasoning; agent must not stop early and must avoid the distractor.

---

## Baseline Results

Run with a deterministic heuristic (fold → DCE cycle, stop when nothing left):

| Task   | Start | End | Removed | Score |
|--------|-------|-----|---------|-------|
| easy   | 2     | 1   | 1       | 0.500 |
| medium | 3     | 1   | 2       | 0.667 |
| hard   | 6     | 3   | 3       | 0.500 |

A strong agent should achieve score ≥ 0.50 on all tasks and 1.0 on easy and medium.

---

## Environment Variables

| Variable        | Description                              | Default |
|-----------------|------------------------------------------|---------|
| `HF_TOKEN`      | Your Hugging Face API key (required)     | —       |
| `API_BASE_URL`  | LLM endpoint URL                         | `https://router.huggingface.co/v1` |
| `MODEL_NAME`    | Model identifier for inference           | `Qwen/Qwen2.5-72B-Instruct` |
| `ENV_BASE_URL`  | Running environment server URL           | `http://localhost:7860` |
| `COMPILER_TASK` | Task to run (`easy` / `medium` / `hard`) | `easy` |

> **Note:** `inference.py` is placed in the root directory of the project as required by the OpenEnv submission spec.

---

## Setup & Usage

### Run locally (no Docker)

```bash
pip install -r requirements.txt

# Run baseline agent
python baseline.py

# Start the API server
python serve.py
```

### Run with Docker

```bash
docker build -t compiler-env .
docker run -p 7860:7860 compiler-env
```

The server will be available at `http://localhost:7860`.

---

## API Endpoints

| Method | Endpoint | Body                        | Description                    |
|--------|----------|-----------------------------|--------------------------------|
| POST   | `/reset` | `{"task_name": "easy"}`     | Reset env, returns observation |
| POST   | `/step`  | `{"pass_name": "const_fold"}` | Apply pass, returns step result |
| GET    | `/state` | —                           | Current program state          |
| GET    | `/tasks` | —                           | List available tasks           |
| GET    | `/health`| —                           | Health check                   |

### Example session

```bash
# Reset to medium task
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_name": "medium"}'

# Apply const_fold
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"pass_name": "const_fold"}'

# Apply dead_code_elim
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"pass_name": "dead_code_elim"}'

# Stop cleanly
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"pass_name": "stop"}'
```

---

## Run Inference Agent

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_BASE_URL=http://localhost:7860
export COMPILER_TASK=easy

python inference.py
```

### Expected stdout format

```
[START] task=easy env=compiler-optimization-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=const_fold reward=-0.05 done=false error=null
[STEP] step=2 action=const_fold reward=-0.05 done=false error=null
[STEP] step=3 action=dead_code_elim reward=0.95 done=false error=null
[STEP] step=4 action=stop reward=1.00 done=true error=null
[END] success=true steps=4 score=0.500 rewards=-0.05,-0.05,0.95,1.00
```

---

## Project Structure

```
.
├── Dockerfile            # Container definition (port 7860)
├── requirements.txt      # Python dependencies
├── openenv.yaml          # OpenEnv spec config
├── serve.py              # FastAPI server (reset / step / state)
├── baseline.py           # Deterministic heuristic baseline
├── inference.py          # LLM agent following OpenEnv stdout spec
└── env/
    ├── env.py            # CompilerEnv — core environment logic
    ├── models.py         # Pydantic models (Observation, Action, etc.)
    ├── passes.py         # const_fold, dead_code_elim, noop transforms
    ├── tasks.py          # Task definitions (easy / medium / hard)
    └── graders.py        # grade() — normalized score in [0, 1]
```