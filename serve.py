from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from env.env import CompilerEnv, Action
from env.tasks import get_tasks
import uvicorn

app = FastAPI(title="Compiler Optimization Environment")

# ✅ ADD THIS
@app.get("/")
def home():
    return {
        "message": "MetaMalloc RL Query Optimizer is running",
        "available_endpoints": ["/reset", "/step", "/tasks", "/health"]
    }

# -----------------------------
# Load tasks once at startup
# -----------------------------
_tasks_list = get_tasks()
TASKS = {t["name"]: t for t in _tasks_list}
DEFAULT_TASK = _tasks_list[0]["name"]

env: CompilerEnv = CompilerEnv(TASKS[DEFAULT_TASK])


# -----------------------------
# Request Models
# -----------------------------
class ResetRequest(BaseModel):
    task_name: str = DEFAULT_TASK


# -----------------------------
# API Endpoints
# -----------------------------
@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    global env

    task_name = body.task_name or DEFAULT_TASK

    if task_name not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_name}'. Available: {list(TASKS.keys())}",
        )

    env = CompilerEnv(TASKS[task_name])
    obs = env.reset()
    return obs.dict()


@app.post("/step")
def step(action: Action):
    try:
        obs, reward, done, info = env.step(action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state()


@app.get("/tasks")
def list_tasks():
    return {"tasks": list(TASKS.keys())}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=7861)