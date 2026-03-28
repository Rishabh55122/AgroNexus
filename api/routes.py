"""
All OpenEnv API endpoints for the Precision Agriculture environment.

Endpoints:
    POST /reset         — Reset environment, return first observation
    POST /step          — Submit action, advance one day
    GET  /state         — Return full internal state
    GET  /tasks         — List tasks and action schema
    POST /grader        — Return grader score after episode
    POST /baseline      — Run baseline policy, return scores
    POST /simulate      — Run full episode, return trajectory
"""
import os
import json
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from env.environment import AgriculturalEnvironment, Action, TASK_CONFIGS
from env.graders import GraderFactory

router = APIRouter()

# ─────────────────────────────────────────────
# Global environment registry
# One env instance per task_id — persists across requests
# ─────────────────────────────────────────────
_envs: dict[str, AgriculturalEnvironment] = {}


def _get_env(task_id: str) -> AgriculturalEnvironment:
    """Returns environment for task_id, creates if not exists."""
    if task_id not in TASK_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. "
                   f"Valid: {list(TASK_CONFIGS.keys())}"
        )
    if task_id not in _envs:
        _envs[task_id] = AgriculturalEnvironment(task_id)
    return _envs[task_id]


# ─────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_1_easy"


class StepRequest(BaseModel):
    task_id:     str = "task_1_easy"
    action_type: str
    zone:        Optional[int]   = None
    amount:      Optional[float] = None
    type:        Optional[str]   = None
    days:        Optional[int]   = None
    market:      Optional[str]   = None
    crop:        Optional[str]   = None


class GraderRequest(BaseModel):
    task_id: str = "task_1_easy"


class SimulateRequest(BaseModel):
    task_id: str    = "task_1_easy"
    policy:  str    = "random"    # "random" | "greedy" | "wait"
    seed:    int    = 42


class BaselineRequest(BaseModel):
    task_id: Optional[str] = None   # None = run all 3 tasks


# ─────────────────────────────────────────────
# POST /reset
# ─────────────────────────────────────────────

@router.post("/reset")
def reset(request: ResetRequest):
    """
    Resets the environment for the given task and returns
    the first observation. Call this before every new episode.
    """
    env = _get_env(request.task_id)
    obs = env.reset()
    return {
        "task_id":     request.task_id,
        "observation": obs.model_dump(),
    }


# ─────────────────────────────────────────────
# POST /step
# ─────────────────────────────────────────────

@router.post("/step")
def step(request: StepRequest):
    """
    Submits an action and advances the environment one day.
    Returns observation, reward, done flag, and info dict.
    """
    env = _get_env(request.task_id)

    if env.day == 0:
        raise HTTPException(
            status_code=400,
            detail="Environment not reset. Call POST /reset first."
        )

    try:
        action = Action(
            action_type=request.action_type,
            zone=request.zone,
            amount=request.amount,
            type=request.type,
            days=request.days,
            market=request.market,
            crop=request.crop,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }


# ─────────────────────────────────────────────
# GET /state
# ─────────────────────────────────────────────

@router.get("/state")
def state(task_id: str = "task_1_easy"):
    """
    Returns the full internal state of the environment.
    Used by the OpenEnv validator to inspect the environment.
    """
    env = _get_env(task_id)
    return env.state()


# ─────────────────────────────────────────────
# GET /tasks
# ─────────────────────────────────────────────

@router.get("/tasks")
def list_tasks():
    """
    Returns all available tasks with their configs
    and the full action schema. Required by OpenEnv spec.
    """
    tasks = []
    for task_id, cfg in TASK_CONFIGS.items():
        task_file = f"tasks/task_{cfg['scenario'].split('_')[0]}.json"
        try:
            with open(task_file) as f:
                task_data = json.load(f)
        except Exception:
            task_data = {}

        tasks.append({
            "task_id":       task_id,
            "difficulty":    task_data.get("difficulty", ""),
            "description":   task_data.get("description", ""),
            "episode_days":  cfg["episode_days"],
            "num_zones":     cfg["num_zones"],
            "scenario":      cfg["scenario"],
            "target_score":  [
                task_data.get("target_score_min", 0.0),
                task_data.get("target_score_max", 1.0),
            ],
            "action_schema": {
                "action_type": {
                    "type": "string",
                    "enum": [
                        "irrigate", "fertilize", "pesticide",
                        "harvest", "wait", "sell"
                    ],
                    "required": True,
                },
                "zone":   {"type": "int",    "required": False,
                           "description": "Zone index (0-based)"},
                "amount": {"type": "float",  "required": False,
                           "description": "Litres for irrigate, tons for sell"},
                "type":   {"type": "string", "required": False,
                           "description": "nitrogen | phosphorus | potassium"},
                "days":   {"type": "int",    "required": False,
                           "description": "Days to wait"},
                "market": {"type": "string", "required": False,
                           "description": "local | export"},
                "crop":   {"type": "string", "required": False,
                           "description": "wheat | corn | soy"},
            },
        })
    return {"tasks": tasks}


# ─────────────────────────────────────────────
# POST /grader
# ─────────────────────────────────────────────

@router.post("/grader")
def grade(request: GraderRequest):
    """
    Returns the grader score for the current episode.
    Episode must be complete (done=True) before calling.
    """
    env = _get_env(request.task_id)

    if not env.done:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Episode not finished for '{request.task_id}'. "
                f"Current day: {env.day} / {env.episode_days}. "
                "Keep stepping until done=True."
            )
        )

    result = env.grade()
    return {
        "task_id": request.task_id,
        "grader_result": result,
    }


# ─────────────────────────────────────────────
# POST /simulate
# ─────────────────────────────────────────────

@router.post("/simulate")
def simulate(request: SimulateRequest):
    """
    Runs a complete episode using a built-in policy.
    Returns full reward trajectory and final grader score.
    Useful for judges to quickly test the environment.

    Policies:
        random  — random valid action each step
        greedy  — irrigate when dry, harvest when ready, else wait
        wait    — always wait (baseline floor)
    """
    import random
    rng = random.Random(request.seed)

    env = _get_env(request.task_id)
    obs = env.reset()

    trajectory = []
    done       = False
    step_num   = 0

    while not done:
        action = _pick_action(obs, request.policy, rng)
        obs, reward, done, info = env.step(action)
        step_num += 1
        trajectory.append({
            "day":          obs.day,
            "step_reward":  reward.step_reward,
            "episode_reward": reward.episode_reward,
            "action":       action.model_dump(),
            "done":         done,
        })

    grader_result = env.grade()

    return {
        "task_id":        request.task_id,
        "policy":         request.policy,
        "total_steps":    step_num,
        "final_reward":   reward.episode_reward,
        "grader_result":  grader_result,
        "trajectory":     trajectory,
    }


def _pick_action(obs, policy: str, rng) -> Action:
    """Picks action based on policy name."""
    import random

    if policy == "wait":
        return Action(action_type="wait", days=1)

    elif policy == "random":
        action_types = [
            "irrigate", "fertilize", "pesticide",
            "harvest", "wait", "sell"
        ]
        action_type = rng.choice(action_types)
        zone = rng.randint(0, max(0, len(obs.zones) - 1))
        return Action(
            action_type=action_type,
            zone=zone,
            amount=rng.uniform(5.0, 20.0),
            market=rng.choice(["local", "export"]),
            crop=rng.choice(["wheat", "corn", "soy"]),
        )

    elif policy == "greedy":
        # Priority 1: harvest any ready zone
        for i, zone in enumerate(obs.zones):
            if zone.growth_stage == 4 and not zone.is_harvested:
                return Action(action_type="harvest", zone=i)

        # Priority 2: apply pesticide if pest risk high
        for i, zone in enumerate(obs.zones):
            if zone.pest_risk > 0.6 and not zone.is_harvested:
                return Action(action_type="pesticide", zone=i)

        # Priority 3: irrigate driest zone
        driest_idx = min(
            range(len(obs.zones)),
            key=lambda i: obs.zones[i].soil_moisture
        )
        if obs.zones[driest_idx].soil_moisture < 0.3:
            return Action(
                action_type="irrigate",
                zone=driest_idx,
                amount=10.0,
            )

        # Default: wait
        return Action(action_type="wait", days=1)

    return Action(action_type="wait", days=1)


# ─────────────────────────────────────────────
# POST /baseline
# ─────────────────────────────────────────────

@router.post("/baseline")
def baseline(request: BaselineRequest):
    """
    Runs the greedy baseline policy across all 3 tasks
    (or a single task if task_id specified).
    Returns reproducible baseline scores for the submission.
    Required by OpenEnv spec.
    """
    task_ids = (
        [request.task_id]
        if request.task_id
        else list(TASK_CONFIGS.keys())
    )

    results = {}
    for task_id in task_ids:
        start_time = time.time()

        env  = AgriculturalEnvironment(task_id)
        obs  = env.reset()
        done = False

        import random
        rng = random.Random(42)

        while not done:
            action       = _pick_action(obs, "greedy", rng)
            obs, reward, done, info = env.step(action)

        grader_result = env.grade()
        elapsed       = round(time.time() - start_time, 2)

        results[task_id] = {
            "final_score":    grader_result["final_score"],
            "episode_reward": reward.episode_reward,
            "grader_result":  grader_result,
            "elapsed_seconds": elapsed,
            "policy":         "greedy",
            "seed":           42,
        }

    return {
        "baseline_scores": results,
        "note": (
            "Greedy policy: harvest when ready, "
            "pesticide when risk > 0.6, irrigate when dry."
        ),
    }
