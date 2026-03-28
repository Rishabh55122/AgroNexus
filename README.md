---
title: AgroNexus OpenEnv
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
---
# AgroNexus — AI Precision Agriculture Decision Agent

An OpenEnv-compliant reinforcement learning environment where an AI agent
manages a simulated farm across a 90-day growing season. The agent makes
daily decisions — irrigate, fertilize, apply pesticide, harvest, sell —
to maximize crop yield and profit under real-world constraints.

## Environment Description

The agent manages a 4×4 grid of farm zones across a growing season.
Each zone has its own crop type, soil moisture, crop health, pest risk,
and growth stage. The agent must balance water budgets, pest outbreaks,
market timing, and crop health to maximize yield and profit.

### What makes this unique

- **Zone interdependency** — pest risk spreads 30% into adjacent zones
  daily if untreated. Over-irrigation causes water runoff into neighbors.
  Failed zones raise stress in adjacent zones.
- **Delayed rewards** — a bad irrigation decision on day 15 shows up as
  crop loss on day 45. Genuine long-horizon credit assignment problem.
- **Stochastic weather** — rainy / dry / heatwave / frost events with
  scenario-weighted probabilities. Drought activates mid-season.
- **Dynamic market prices** — prices fluctuate ±15% daily. Export market
  opens every 7 days at 20% premium. Crisis scenario triggers a 30% crash.
- **State memory** — last 5 days of observations included. Agent can
  reason across time, not just react to current snapshot.

## Action Space

All actions are flat JSON dicts with `action_type` as the key:

```json
{ "action_type": "irrigate",  "zone": 2, "amount": 10.0 }
{ "action_type": "fertilize", "zone": 1, "type": "nitrogen" }
{ "action_type": "pesticide", "zone": 3 }
{ "action_type": "harvest",   "zone": 0 }
{ "action_type": "wait",      "days": 1 }
{ "action_type": "sell",      "amount": 50.0, "crop": "wheat", "market": "export" }
```

## Observation Space

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

## Tasks

| Task | Difficulty | Days | Zones | Scenario | Target Score |
|------|-----------|------|-------|----------|-------------|
| task_1_easy | Easy | 30 | 2 | Normal | 0.7 – 1.0 |
| task_2_medium | Medium | 60 | 4 | Drought | 0.4 – 0.7 |
| task_3_hard | Hard | 90 | 16 | Crisis | 0.2 – 0.5 |

## Setup

### Local

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t agronexus .
docker run -p 8000:8000 agronexus
```

### Baseline

```bash
python baseline.py
python baseline.py --task task_1_easy
python baseline.py --policy llm --api http://localhost:8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /reset | Reset environment, return first observation |
| POST | /step | Submit action, advance one day |
| GET | /state | Return full internal state |
| GET | /tasks | List all tasks and action schema |
| POST | /grader | Return grader score after episode |
| POST | /baseline | Run baseline policy, return all scores |
| POST | /simulate | Run full episode, return trajectory |

## Baseline Scores (greedy policy, seed=42)

| Task | Score | Episode Reward |
|------|-------|----------------|
| task_1_easy | TBD | TBD |
| task_2_medium | TBD | TBD |
| task_3_hard | TBD | TBD |

*Run `python baseline.py` to generate scores.*

## Project Structure

```
agronexus/
├── env/
│   ├── environment.py
│   ├── weather.py
│   ├── farm.py
│   ├── grid.py
│   ├── market.py
│   └── graders.py
├── tasks/
│   ├── task_easy.json
│   ├── task_medium.json
│   └── task_hard.json
├── api/
│   ├── main.py
│   └── routes.py
├── baseline.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

## License

MIT
