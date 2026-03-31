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
    task_id:     str             = "task_1_easy"
    action_type: str             = "wait"
    zone:        Optional[int]   = None
    amount:      Optional[float] = None
    type:        Optional[str]   = None
    days:        Optional[int]   = 1
    market:      Optional[str]   = None
    crop:        Optional[str]   = None

class GraderRequest(BaseModel):
    task_id: str = "task_1_easy"

class SimulateRequest(BaseModel):
    task_id: str = "task_1_easy"
    policy:  str = "greedy"
    seed:    int = 42

class BaselineRequest(BaseModel):
    task_id: Optional[str] = None

# ─────────────────────────────────────────────
# POST /reset
# ─────────────────────────────────────────────

@router.post("/reset")
def reset(request: ResetRequest = None):
    """Reset environment. Works with empty body — defaults to task_1_easy."""
    if request is None:
        request = ResetRequest()
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
def step(request: StepRequest = None):
    """Submit action. Works with empty body — defaults to wait action."""
    if request is None:
        request = StepRequest()
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
    Returns all available tasks with full descriptions,
    configs and complete action schema.
    Required by OpenEnv spec.
    """
    import json as _json

    task_file_map = {
        "task_1_easy":   "tasks/task_easy.json",
        "task_2_medium": "tasks/task_medium.json",
        "task_3_hard":   "tasks/task_hard.json",
    }

    tasks = []
    for task_id, cfg in TASK_CONFIGS.items():
        task_data = {}
        filepath = task_file_map.get(task_id, "")
        try:
            with open(filepath) as f:
                task_data = _json.load(f)
        except Exception as e:
            print(f"Warning: could not read {filepath}: {e}")

        tasks.append({
            "task_id":      task_id,
            "difficulty":   task_data.get(
                "difficulty",
                cfg.get("scenario", "unknown")
            ),
            "description":  task_data.get(
                "description",
                f"{task_id} — {cfg.get('scenario','normal')} scenario, "
                f"{cfg.get('episode_days',30)} days, "
                f"{cfg.get('num_zones',2)} zones"
            ),
            "episode_days":  cfg["episode_days"],
            "num_zones":     cfg["num_zones"],
            "scenario":      cfg["scenario"],
            "budget":        cfg.get("budget", 1000.0),
            "water_budget":  cfg.get("water_budget", 500.0),
            "target_score": [
                task_data.get("target_score_min", 0.0),
                task_data.get("target_score_max", 1.0),
            ],
            "success_criteria": task_data.get("success_criteria", []),
            "action_schema": {
                "action_type": {
                    "type":        "string",
                    "enum":        [
                        "irrigate", "fertilize", "pesticide",
                        "harvest", "wait", "sell"
                    ],
                    "required":    True,
                    "description": (
                        "Action to perform this turn. "
                        "irrigate=add water to zone, "
                        "fertilize=boost crop health, "
                        "pesticide=reduce pest risk, "
                        "harvest=collect crop at stage 4 only, "
                        "wait=skip turn, "
                        "sell=sell harvested crop for revenue"
                    ),
                },
                "zone": {
                    "type":        "int",
                    "required":    False,
                    "description": (
                        f"Zone index 0-based "
                        f"(0 to {cfg.get('num_zones',2)-1}). "
                        "Required for irrigate, fertilize, "
                        "pesticide, harvest."
                    ),
                },
                "amount": {
                    "type":        "float",
                    "required":    False,
                    "description": (
                        "Litres of water for irrigate (5.0-50.0). "
                        "Tons of crop for sell (1.0-100.0)."
                    ),
                },
                "type": {
                    "type":        "string",
                    "required":    False,
                    "description": (
                        "Fertilizer type: "
                        "nitrogen (boosts health +0.05) | "
                        "phosphorus | potassium"
                    ),
                },
                "days": {
                    "type":        "int",
                    "required":    False,
                    "description": "Number of days to wait. Default 1.",
                },
                "market": {
                    "type":        "string",
                    "required":    False,
                    "description": (
                        "local = always open, base price. "
                        "export = opens every 7 days, "
                        "20% price premium."
                    ),
                },
                "crop": {
                    "type":        "string",
                    "required":    False,
                    "description": "wheat | corn | soy",
                },
            },
        })

    return {"tasks": tasks}


# ─────────────────────────────────────────────
# POST /grader
# ─────────────────────────────────────────────

@router.post("/grader")
def grade(request: GraderRequest = None):
    """Grade episode. Works with empty body — defaults to task_1_easy."""
    if request is None:
        request = GraderRequest()
    env = _get_env(request.task_id)
    if not env.done:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Episode not finished for '{request.task_id}'. "
                f"Current day: {env.day} / {env.episode_days}."
            )
        )
    result = env.grade()
    return {
        "task_id":       request.task_id,
        "grader_result": result,
    }


# ─────────────────────────────────────────────
# POST /simulate
# ─────────────────────────────────────────────

@router.post("/simulate")
def simulate(request: SimulateRequest = None):
    """Run full episode. Works with empty body — defaults to task_1_easy greedy."""
    if request is None:
        request = SimulateRequest()
    import random
    rng = random.Random(request.seed)
    env = AgriculturalEnvironment(request.task_id)
    obs = env.reset()
    trajectory = []
    done       = False
    step_num   = 0
    while not done:
        action = _pick_action(obs, request.policy, rng)
        obs, reward, done, info = env.step(action)
        step_num += 1
        trajectory.append({
            "day":            obs.day,
            "step_reward":    reward.step_reward,
            "episode_reward": reward.episode_reward,
            "action":         action.model_dump(),
            "done":           done,
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
def baseline(request: BaselineRequest = None):
    """Run baseline across all tasks. Works with empty body."""
    if request is None:
        request = BaselineRequest()
    import time, random
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
        rng  = random.Random(42)
        while not done:
            action = _pick_action(obs, "greedy", rng)
            obs, reward, done, info = env.step(action)
        grader_result = env.grade()
        elapsed = round(time.time() - start_time, 2)
        results[task_id] = {
            "final_score":     grader_result["final_score"],
            "episode_reward":  reward.episode_reward,
            "grader_result":   grader_result,
            "elapsed_seconds": elapsed,
            "policy":          "greedy",
            "seed":            42,
        }
    return {
        "baseline_scores": results,
        "note": "Greedy policy — seed 42 — reproducible.",
    }
