from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from env.env import CompilerEnv, Action
from env.tasks import get_tasks
import uvicorn

app = FastAPI(title="Compiler Optimization Environment")

# Build task lookup once at startup
_tasks_list = get_tasks()
TASKS: dict = {t["name"]: t for t in _tasks_list}

# One global env instance (stateful per session)
env: CompilerEnv = CompilerEnv(TASKS["easy"])


class ResetRequest(BaseModel):
    task_name: str = "easy"


@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    global env
    if body.task_name not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{body.task_name}'. Available: {list(TASKS)}")
    env = CompilerEnv(TASKS[body.task_name])
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
    uvicorn.run("serve:app", host="0.0.0.0", port=7860, reload=False)
