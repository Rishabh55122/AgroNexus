"""
AgroNexus — Baseline Inference Script

Runs a greedy policy against all 3 tasks using the OpenAI API.
Reads OPENAI_API_KEY from environment variables.
Produces reproducible baseline scores.

Usage:
    export OPENAI_API_KEY=your_key_here
    python baseline.py

    # Run single task:
    python baseline.py --task task_1_easy

    # Run against live API server:
    python baseline.py --api http://localhost:8000
"""
import os
import json
import time
import argparse
import random
from typing import Optional

BASE_URL = None   # set via --api flag, else runs locally


def get_args():
    parser = argparse.ArgumentParser(
        description="AgroNexus Baseline"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Run single task. Options: task_1_easy, task_2_medium, task_3_hard"
    )
    parser.add_argument(
        "--api",
        type=str,
        default=None,
        help="Base URL of running API server e.g. http://localhost:8000"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="greedy",
        choices=["greedy", "llm", "random"],
        help="Policy to run: greedy (default), llm (uses OpenAI), random"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# LOCAL MODE — imports environment directly
# ─────────────────────────────────────────────

def run_local(task_id: str, policy: str, seed: int) -> dict:
    """
    Runs episode locally by importing AgriculturalEnvironment directly.
    Does not require the API server to be running.
    """
    from env.environment import AgriculturalEnvironment, Action

    print(f"  Running {task_id} locally with {policy} policy...")
    rng = random.Random(seed)

    env  = AgriculturalEnvironment(task_id)
    obs  = env.reset()
    done = False
    step = 0

    while not done:
        if policy == "llm":
            action = get_llm_action(obs, task_id)
        elif policy == "greedy":
            action = get_greedy_action(obs, rng)
        else:
            action = get_random_action(obs, rng)

        obs, reward, done, info = env.step(action)
        step += 1

        if step % 10 == 0:
            print(
                f"    Day {obs.day}: "
                f"reward={reward.step_reward:.3f} "
                f"budget=${obs.budget_left:.2f}"
            )

    result = env.grade()
    return {
        "task_id":        task_id,
        "policy":         policy,
        "seed":           seed,
        "total_steps":    step,
        "final_score":    result["final_score"],
        "episode_reward": reward.episode_reward,
        "grader_result":  result,
    }


# ─────────────────────────────────────────────
# API MODE — hits running FastAPI server
# ─────────────────────────────────────────────

def run_via_api(task_id: str, policy: str, seed: int, base_url: str) -> dict:
    """
    Runs episode via the live FastAPI server.
    Requires server to be running at base_url.
    """
    import requests as req

    print(f"  Running {task_id} via API at {base_url}...")

    # Reset
    resp = req.post(f"{base_url}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs  = resp.json()["observation"]
    done = False
    step = 0
    rng  = random.Random(seed)
    last_reward = None

    while not done:
        action = _pick_api_action(obs, policy, rng)
        resp   = req.post(f"{base_url}/step", json={
            "task_id": task_id,
            **action,
        })
        resp.raise_for_status()
        data        = resp.json()
        obs         = data["observation"]
        last_reward = data["reward"]
        done        = data["done"]
        step       += 1

        if step % 10 == 0:
            print(
                f"    Day {obs['day']}: "
                f"reward={last_reward['step_reward']:.3f} "
                f"budget=${obs['budget_left']:.2f}"
            )

    # Grade
    resp = req.post(f"{base_url}/grader", json={"task_id": task_id})
    resp.raise_for_status()
    result = resp.json()["grader_result"]

    return {
        "task_id":        task_id,
        "policy":         policy,
        "seed":           seed,
        "total_steps":    step,
        "final_score":    result["final_score"],
        "episode_reward": last_reward["episode_reward"] if last_reward else 0.0,
        "grader_result":  result,
    }


# ─────────────────────────────────────────────
# GREEDY POLICY
# ─────────────────────────────────────────────

def get_greedy_action(obs, rng):
    """
    Greedy policy:
    1. Harvest any zone at growth_stage=4
    2. Apply pesticide if pest_risk > 0.6
    3. Irrigate driest zone if moisture < 0.3
    4. Wait
    """
    from env.environment import Action

    zones = obs.zones

    # Priority 1: harvest ready zones
    for i, z in enumerate(zones):
        if z.growth_stage == 4 and not z.is_harvested:
            return Action(action_type="harvest", zone=i)

    # Priority 2: pesticide high risk zones
    for i, z in enumerate(zones):
        if z.pest_risk > 0.6 and not z.is_harvested:
            return Action(action_type="pesticide", zone=i)

    # Priority 3: irrigate driest non-harvested zone
    candidates = [
        (i, z) for i, z in enumerate(zones)
        if not z.is_harvested
    ]
    if candidates:
        driest_i, driest_z = min(
            candidates,
            key=lambda x: x[1].soil_moisture
        )
        if driest_z.soil_moisture < 0.3:
            return Action(
                action_type="irrigate",
                zone=driest_i,
                amount=10.0,
            )

    # Priority 4: sell on export days (every 7 days)
    if obs.day % 7 == 0:
        return Action(
            action_type="sell",
            amount=50.0,
            crop="wheat",
            market="export",
        )

    return Action(action_type="wait", days=1)


def get_random_action(obs, rng):
    """Random valid action."""
    from env.environment import Action

    action_types = [
        "irrigate", "fertilize", "pesticide",
        "harvest", "wait", "sell"
    ]
    action_type = rng.choice(action_types)
    zone = rng.randint(0, max(0, len(obs.zones) - 1))
    return Action(
        action_type=action_type,
        zone=zone,
        amount=round(rng.uniform(5.0, 20.0), 1),
        market=rng.choice(["local", "export"]),
        crop=rng.choice(["wheat", "corn", "soy"]),
    )


def _pick_api_action(obs: dict, policy: str, rng) -> dict:
    """Builds action dict for API mode."""
    zones = obs.get("zones", [])

    if policy == "greedy":
        # Harvest ready
        for i, z in enumerate(zones):
            if z["growth_stage"] == 4 and not z["is_harvested"]:
                return {"action_type": "harvest", "zone": i}
        # Pesticide
        for i, z in enumerate(zones):
            if z["pest_risk"] > 0.6 and not z["is_harvested"]:
                return {"action_type": "pesticide", "zone": i}
        # Irrigate driest
        candidates = [
            (i, z) for i, z in enumerate(zones)
            if not z["is_harvested"]
        ]
        if candidates:
            di, dz = min(candidates, key=lambda x: x[1]["soil_moisture"])
            if dz["soil_moisture"] < 0.3:
                return {"action_type": "irrigate", "zone": di, "amount": 10.0}
        # Sell on export day
        if obs.get("day", 0) % 7 == 0:
            return {
                "action_type": "sell",
                "amount": 50.0,
                "crop": "wheat",
                "market": "export",
            }
        return {"action_type": "wait", "days": 1}

    elif policy == "random":
        action_types = [
            "irrigate", "fertilize", "pesticide",
            "harvest", "wait", "sell"
        ]
        zone = rng.randint(0, max(0, len(zones) - 1))
        return {
            "action_type": rng.choice(action_types),
            "zone": zone,
            "amount": round(rng.uniform(5.0, 20.0), 1),
            "market": rng.choice(["local", "export"]),
            "crop": rng.choice(["wheat", "corn", "soy"]),
        }

    return {"action_type": "wait", "days": 1}


# ─────────────────────────────────────────────
# LLM POLICY — uses OpenAI API
# ─────────────────────────────────────────────

def get_llm_action(obs, task_id: str):
    """
    LLM policy — sends observation to GPT and parses action.
    Falls back to greedy if API call fails or parse error.
    """
    from env.environment import Action
    import random

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("    WARNING: OPENAI_API_KEY not set, falling back to greedy")
        return get_greedy_action(obs, random.Random(42))

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Build observation summary for LLM
        zones_summary = []
        for i, z in enumerate(obs.zones):
            zones_summary.append(
                f"Zone {i} ({z.zone_id}, {z.crop_type}): "
                f"health={z.crop_health:.2f}, "
                f"moisture={z.soil_moisture:.2f}, "
                f"pest={z.pest_risk:.2f}, "
                f"stage={z.growth_stage}/4, "
                f"days_to_ready={z.days_to_ready}, "
                f"harvested={z.is_harvested}"
            )

        prompt = f"""You are an AI farm manager. Choose the best action for today.

Day: {obs.day} / {obs.season_phase}
Budget: ${obs.budget_left:.2f}
Weather: {obs.weather.event}, rainfall={obs.weather.rainfall_mm}mm, temp={obs.weather.temperature:.1f}C

Zones:
{chr(10).join(zones_summary)}

Available actions:
- irrigate: zone (int), amount (float, litres) — costs $0.10/litre
- fertilize: zone (int), type (nitrogen/phosphorus/potassium) — costs $25
- pesticide: zone (int) — costs $40
- harvest: zone (int) — only works at stage 4
- wait: days (int)
- sell: amount (float), crop (wheat/corn/soy), market (local/export)

Respond with ONLY a JSON object. Example:
{{"action_type": "irrigate", "zone": 2, "amount": 10.0}}

Choose the single best action right now."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1,
        )

        content = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        content = content.replace("```json", "").replace("```", "").strip()
        action_dict = json.loads(content)

        return Action(**action_dict)

    except Exception as e:
        print(f"    LLM error: {e} — falling back to greedy")
        return get_greedy_action(obs, random.Random(42))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args    = get_args()
    task_ids = (
        [args.task] if args.task
        else ["task_1_easy", "task_2_medium", "task_3_hard"]
    )

    print("=" * 60)
    print("AgroNexus — Baseline")
    print(f"Policy: {args.policy} | Seed: {args.seed}")
    if args.api:
        print(f"Mode:   API ({args.api})")
    else:
        print("Mode:   Local (direct import)")
    print("=" * 60)

    all_results = {}

    for task_id in task_ids:
        print(f"\n[{task_id}]")
        start = time.time()

        try:
            if args.api:
                result = run_via_api(
                    task_id, args.policy, args.seed, args.api
                )
            else:
                result = run_local(
                    task_id, args.policy, args.seed
                )

            elapsed = round(time.time() - start, 2)
            result["elapsed_seconds"] = elapsed
            all_results[task_id] = result

            print(f"  Final score:    {result['final_score']}")
            print(f"  Episode reward: {result['episode_reward']:.4f}")
            print(f"  Elapsed:        {elapsed}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[task_id] = {"error": str(e)}

    print("\n" + "=" * 60)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 60)
    for task_id, result in all_results.items():
        if "error" in result:
            print(f"  {task_id}: ERROR — {result['error']}")
        else:
            print(
                f"  {task_id}: "
                f"score={result['final_score']} "
                f"reward={result['episode_reward']:.4f} "
                f"({result['elapsed_seconds']}s)"
            )

    # Save results to file
    output_path = "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
