---
title: AgroNexus OpenEnv
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
---

# AgroNexus / Precision Farming
> An algorithmic agriculture decision engine and OpenEnv simulator

AgroNexus is a reinforcement learning sandbox built to simulate the punishing realities of modern, multi-zone precision farming. You manage a 90-day growing season by balancing rigid water budgets, timing crop harvests to volatile market prices, and fighting ruthless pest contagions before they wipe out the grid.

Included is a complete FastAPI simulation backend and a custom ops-dashboard for manual farming, task observation, and algorithmic grading.

## The Simulation

Agents manage up to a 16-zone farm grid. Every zone maintains local state (soil moisture, crop health, pest risk, growth stage). Actions are taken daily. 

**Key mechanical constraints:**
- **Spatial Contagion:** Pest risks spread at a 30% transmission rate to adjacent zones daily. Over-irrigation causes water runoff into neighbors, drowning root systems.
- **Credit Assignment:** Delayed rewards are strictly enforced. Poor irrigation execution on day 15 surfaces as terminal crop loss on day 45.
- **Stochastic Events:** Weather is procedural (rainy, dry, heatwaves, frost). Mid-season evaluation triggers probability-weighted scenario shifts (e.g. severe droughts).
- **Market Volatility:** Crop prices fluctuate ±15% daily. High-yield export markets open only every 7 days, carrying a 20% premium. Crisis scenarios trigger a sudden 30% price crash.

## API Space

State and actions are evaluated via flat JSON dicts. 

### Actions
Supported actions: `irrigate`, `fertilize`, `pesticide`, `harvest`, `wait`, `sell`.

```json
{ "action_type": "irrigate",  "zone": 2, "amount": 10.0 }
{ "action_type": "sell",      "amount": 50.0, "crop": "wheat", "market": "export" }
```

### Observations
The environment returns current day statistics alongside a rolling 5-day historical memory.

```json
{
  "day": 42,
  "season_phase": "growing",
  "budget_left": 742.50,
  "market_prices": { "wheat": { "local_price": 2.81, "export_price": 3.37, "export_open": false } },
  "weather": { "event": "dry", "rainfall_mm": 0.0, "temperature": 33.2, "humidity": 0.24, "frost": false },
  "zones": [
    {
      "zone_id": "A1", "crop_type": "wheat",
      "soil_moisture": 0.42, "crop_health": 0.88,
      "growth_stage": 2, "pest_risk": 0.15,
      "days_to_ready": 18, "yield_potential": 0.91,
      "is_harvested": false
    }
  ],
  "history": [ "...last 5 days..." ]
}
```

## Scenarios

| Core Task | Difficulty | Duration | Scale | Condition | Target Score |
|---|---|---|---|---|---|
| `task_1_easy` | Easy | 30 days | 2 zones | Normal | 0.70 – 1.00 |
| `task_2_medium` | Medium | 60 days | 4 zones | Drought | 0.40 – 0.70 |
| `task_3_hard` | Hard | 90 days | 16 zones | Crisis | 0.20 – 0.50 |

## Quickstart

### Local Development
Requires Python 3.11+.

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Then navigate to `http://localhost:8000/` for the interactive tractor dashboard.

### Docker
```bash
docker build -t agronexus .
docker run -p 8000:8000 agronexus
```

### Programmatic Baseline
```bash
# Run greedy policy evaluation
python baseline.py --task task_1_easy

# Run LLM-driven policy (requires OPENAI_API_KEY)
python baseline.py --policy llm --api http://localhost:8000
```

## Structure

```text
├── api/             # FastAPI routing and entrypoints
├── env/             # Core simulation logic (weather, market, farm grid)
├── server/          # OpenEnv required multi-mode app hook
├── tasks/           # OpenEnv scenario definitions
├── ui/              # Static ops-dashboard HTML/Vanilla JS 
├── baseline.py      # Starter policies implementation
├── inference.py     # OpenEnv standard evaluation script
├── openenv.yaml     # Metadata
└── Dockerfile       # Deployment config
```

## Endpoints

| Resource | Method | Description |
|---|---|---|
| `/reset` | `POST` | Reset farm environment to day 0. Returns first observation. |
| `/step` | `POST` | Submit a farming action. Advances the engine by N days. |
| `/state` | `GET` | Dump the full internal farm engine state. |
| `/tasks` | `GET` | List all config rules, tasks, and the action schema. |
| `/simulate` | `POST` | Run a full season head-to-tail and return trajectory logs. |

---

License: MIT
