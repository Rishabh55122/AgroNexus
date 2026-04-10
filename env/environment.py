import json
from typing import Optional, Literal
from pydantic import BaseModel, Field
from env.farm import CropType, GrowthStage
from env.weather import WeatherEngine, WeatherScenario
from env.grid import FarmGrid
from env.market import MarketEngine
from env.graders import EpisodeLog, GraderFactory

class WeatherObservation(BaseModel):
    """Weather data visible to the agent each step."""
    event:            str
    rainfall_mm:      float
    temperature:      float
    humidity:         float
    frost:            bool
    rain_probability: float

class ZoneObservation(BaseModel):
    """Single zone state visible to the agent each step."""
    zone_id:         str
    crop_type:       str
    soil_moisture:   float
    crop_health:     float
    growth_stage:    int
    pest_risk:       float
    days_to_ready:   int
    yield_potential: float
    is_harvested:    bool

class Observation(BaseModel):
    """
    Full observation returned by step() and reset().
    This is what the agent sees every turn.
    """
    day:            int
    season_phase:   Literal["planting", "growing", "harvest"]
    budget_left:    float
    market_prices:  dict
    weather:        WeatherObservation
    zones:          list[ZoneObservation]
    history:        list[dict] = Field(default_factory=list)

class Action(BaseModel):
    """
    Action submitted by the agent each step.
    action_type must be one of the 6 valid types.
    """
    action_type: Literal[
        "irrigate",
        "fertilize",
        "pesticide",
        "harvest",
        "wait",
        "sell",
    ]
    zone:    Optional[int]   = None   # zone index (0-based)
    amount:  Optional[float] = None   # water litres or sell amount
    type:    Optional[str]   = None   # fertilizer type: "nitrogen"|"phosphorus"|"potassium"
    days:    Optional[int]   = None   # for wait()
    market:  Optional[str]   = None   # "local" | "export"
    crop:    Optional[str]   = None   # crop type for sell()

class Reward(BaseModel):
    """Reward returned by step() each turn."""
    step_reward:         float   # immediate shaping signal
    episode_reward:      float   # cumulative so far
    yield_component:     float
    health_component:    float
    efficiency_component:float
    done:                bool
    info:                dict = Field(default_factory=dict)

TASK_CONFIGS = {
    "task_1_easy": {
        "num_zones":     2,
        "episode_days":  30,
        "budget":        500.0,
        "water_budget":  200.0,
        "scenario":      "normal",
        "crops":         ["wheat"],
        "seed":          42,
    },
    "task_2_medium": {
        "num_zones":     4,
        "episode_days":  60,
        "budget":        800.0,
        "water_budget":  300.0,
        "scenario":      "drought",
        "crops":         ["wheat", "corn"],
        "seed":          42,
    },
    "task_3_hard": {
        "num_zones":     16,
        "episode_days":  90,
        "budget":        1000.0,
        "water_budget":  500.0,
        "scenario":      "crisis",
        "crops":         ["wheat", "corn", "soy"],
        "seed":          42,
    },
}

class AgriculturalEnvironment:
    """
    Precision Agriculture Decision Environment — OpenEnv compliant.

    An AI agent manages a simulated farm across a growing season.
    Decisions: irrigate, fertilize, apply pesticide, harvest, sell, wait.
    Objective: maximize crop yield and profit under real-world constraints.

    Unique features:
    - 4x4 zone grid with spatial pest spread and water runoff
    - Stochastic weather with drought/heatwave scenarios
    - Dynamic market prices with export market timing
    - Delayed rewards — bad decisions show up days later
    - State memory: last 5 days of observations
    """

    def __init__(self, task_id: str = "task_1_easy"):
        if task_id not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task '{task_id}'. "
                f"Valid: {list(TASK_CONFIGS.keys())}"
            )
        self.task_id = task_id
        self.config  = TASK_CONFIGS[task_id]
        self._setup()

    def _setup(self) -> None:
        """Initializes all sub-engines from task config."""
        cfg = self.config

        self.weather = WeatherEngine(
            scenario=WeatherScenario(cfg["scenario"]),
            seed=cfg["seed"],
        )
        self.grid = FarmGrid(num_zones=cfg["num_zones"])
        self.market = MarketEngine(
            scenario=cfg["scenario"],
            seed=cfg["seed"],
            crops=cfg["crops"],
        )

        self.day             = 0
        self.episode_days    = cfg["episode_days"]
        self.budget          = cfg["budget"]
        self.water_used      = 0.0
        self.water_budget    = cfg["water_budget"]
        self.episode_reward  = 0.0
        self.done            = False

        # State memory — last 5 observations
        self.history: list[dict] = []

        # Episode log for grading
        self.log = EpisodeLog(
            task_id=self.task_id,
            budget_start=cfg["budget"],
            water_budget=cfg["water_budget"],
            zones_total=cfg["num_zones"],
            max_possible_revenue=self._estimate_max_revenue(),
            scenario=cfg["scenario"],
        )

        # Current weather state (set on first step)
        self.current_weather = None

    def reset(self) -> Observation:
        """
        Resets environment to initial state.
        Returns the first observation.
        Required by OpenEnv spec.
        """
        self.weather.reset()
        self.grid.reset()
        self.market.reset()

        self.day            = 0
        self.budget         = self.config["budget"]
        self.water_used     = 0.0
        self.episode_reward = 0.0
        self.done           = False
        self.history        = []

        self.log = EpisodeLog(
            task_id=self.task_id,
            budget_start=self.config["budget"],
            water_budget=self.config["water_budget"],
            zones_total=self.config["num_zones"],
            max_possible_revenue=self._estimate_max_revenue(),
            scenario=self.config["scenario"],
        )

        # Advance one day to get initial state
        self.current_weather = self.weather.step()
        self.market.step()
        self.day = 1

        obs = self._build_observation()
        self.history.append(obs.model_dump())
        return obs

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """
        Advances environment by one day.
        Applies agent action, updates all sub-engines.
        Returns (observation, reward, done, info).
        Required by OpenEnv spec.
        """
        if self.done:
            raise RuntimeError(
                "Episode is done. Call reset() before stepping."
            )

        # --- Apply action ---
        action_info = self._apply_action(action)

        # --- Advance weather and market ---
        # print("DEBUG: weather and market step")
        self.current_weather = self.weather.step()
        self.market.step()

        # --- Update grid ---
        # print("DEBUG: grid step")
        zone_actions = self._get_zone_actions(action)
        self.grid.step(
            self.current_weather.to_dict(),
            zone_actions,
        )

        # --- Advance day ---
        self.day += 1
        self.log.total_days = self.day

        # --- Compute reward ---
        # print("DEBUG: compute reward")
        reward = self._compute_reward(action, action_info)
        self.episode_reward += reward.step_reward
        reward.episode_reward = round(self.episode_reward, 4)

        # --- Check done ---
        self.done = (
            self.day >= self.episode_days or
            self.budget <= 0
        )

        if self.done:
            self._finalize_log()

        # --- Build observation ---
        obs = self._build_observation()

        # --- Update history (keep last 5) ---
        # Create a light dump without history to prevent exponential explosion
        obs_dict = obs.model_dump(exclude={"history"})
        self.history.append(obs_dict)
        if len(self.history) > 5:
            self.history.pop(0)

        reward.done = self.done
        info = {
            "day": self.day,
            "action_info": action_info,
            "budget_left": self.budget,
            "water_used": self.water_used,
        }

        return obs, reward, self.done, info

    def state(self) -> dict:
        """
        Returns full internal state of the environment.
        Required by OpenEnv spec.
        Used by /grader endpoint to inspect episode state.
        """
        return {
            "task_id":        self.task_id,
            "day":            self.day,
            "done":           self.done,
            "budget":         self.budget,
            "water_used":     self.water_used,
            "episode_reward": self.episode_reward,
            "zones":          self.grid.all_zones_dict(),
            "weather":        self.current_weather.to_dict()
                              if self.current_weather else {},
            "market":         self.market.get_current_prices(),
            "history_length": len(self.history),
            "log": {
                "total_revenue":  self.log.total_revenue,
                "zones_failed":   self.log.zones_failed,
                "harvest_events": len(self.log.harvest_events),
                "sell_events":    len(self.log.sell_events),
            },
        }

    def _apply_action(self, action: Action) -> dict:
        """
        Applies the agent's action to the environment.
        Deducts costs from budget. Logs action.
        Returns info dict describing what happened.
        """
        info = {"action_type": action.action_type, "success": True, "reason": ""}

        zone_ids  = list(self.grid.zones.keys())
        zone_id   = zone_ids[action.zone] if action.zone is not None \
                    and action.zone < len(zone_ids) else None

        if action.action_type == "irrigate":
            amount = action.amount or 10.0
            cost   = amount * 0.10          # $0.10 per litre
            if self.budget >= cost:
                self.budget     -= cost
                self.water_used += amount
                self.log.water_used += amount
                info["amount"] = amount
                info["cost"]   = cost
            else:
                info["success"] = False
                info["reason"]  = "insufficient budget"

        elif action.action_type == "fertilize":
            cost = 25.0
            if self.budget >= cost:
                self.budget -= cost
                info["fertilizer_type"] = action.type or "nitrogen"
                info["cost"] = cost
            else:
                info["success"] = False
                info["reason"]  = "insufficient budget"

        elif action.action_type == "pesticide":
            cost = 40.0
            if self.budget >= cost:
                self.budget -= cost
                info["cost"] = cost
            else:
                info["success"] = False
                info["reason"]  = "insufficient budget"

        elif action.action_type == "harvest":
            if zone_id and not self.grid.zones[zone_id].is_harvested:
                zone  = self.grid.zones[zone_id]
                yield_amount = zone.harvest()
                info["zone_id"]      = zone_id
                info["yield_amount"] = yield_amount
                info["growth_stage"] = int(zone.growth_stage)
                self.log.harvest_events.append({
                    "day":          self.day,
                    "zone_id":      zone_id,
                    "growth_stage": int(zone.growth_stage),
                    "yield":        yield_amount,
                })
            else:
                info["success"] = False
                info["reason"]  = "zone already harvested or invalid"

        elif action.action_type == "sell":
            amount   = action.amount or 0.0
            crop     = action.crop or "wheat"
            market   = action.market or "local"
            revenue  = self.market.calculate_revenue(crop, amount, market)
            if revenue > 0:
                self.budget            += revenue
                self.log.total_revenue += revenue
                info["revenue"]  = revenue
                info["crop"]     = crop
                info["market"]   = market
                self.log.sell_events.append({
                    "day":     self.day,
                    "crop":    crop,
                    "amount":  amount,
                    "revenue": revenue,
                })
            else:
                info["success"] = False
                info["reason"]  = "market closed or invalid crop"

        elif action.action_type == "wait":
            info["days"] = action.days or 1

        self.log.actions_taken.append({
            "day":    self.day,
            "action": action.model_dump(),
            "info":   info,
        })

        return info

    def _get_zone_actions(self, action: Action) -> dict:
        """
        Converts Action to the zone_actions dict format
        expected by FarmGrid.step().
        Returns {zone_id: action_type_string}.
        """
        if action.zone is None:
            return {}
        zone_ids = list(self.grid.zones.keys())
        if action.zone >= len(zone_ids):
            return {}
        zone_id = zone_ids[action.zone]
        if action.action_type in ("irrigate","fertilize","pesticide"):
            return {zone_id: action.action_type}
        return {}

    def _compute_reward(
        self,
        action: Action,
        action_info: dict,
    ) -> Reward:
        """
        Computes per-step reward signal.

        Final reward formula:
            yield_quality * market_price * 0.4
            + crop_health_avg * 0.2
            + sustainability * 0.1
            - water_overuse * 0.1
            - pest_damage * 0.1
            - budget_exceeded * 0.1

        Per-step shaping:
            +0.02 if any zone health improved
            -0.02 if pest risk increased without action
        """
        zones = list(self.grid.zones.values())

        # Health component
        avg_health       = sum(z.crop_health for z in zones) / len(zones)
        health_component = round(avg_health * 0.2, 4)

        # Yield component (based on revenue this step)
        step_revenue     = action_info.get("revenue", 0.0)
        avg_market       = sum(
            v["best_price"]
            for v in self.market.get_current_prices().values()
        ) / len(self.market.crops)
        yield_component  = round(
            (step_revenue / max(1.0, self.log.max_possible_revenue))
            * avg_market * 0.4, 4
        )

        # Sustainability component
        water_ratio      = self.water_used / max(1.0, self.water_budget)
        water_overuse    = max(0.0, water_ratio - 1.0) * 0.1
        budget_penalty   = 0.1 if self.budget <= 0 else 0.0
        sustainability   = round(0.1 - water_overuse - budget_penalty, 4)

        # Pest damage
        avg_pest         = sum(z.pest_risk for z in zones) / len(zones)
        pest_damage      = round(avg_pest * 0.1, 4)

        # Per-step shaping signals
        shaping = 0.0
        health_improved = any(
            z.crop_health > 0.5 and not z.is_harvested
            for z in zones
        )
        if health_improved:
            shaping += 0.02
        if avg_pest > 0.3 and action.action_type != "pesticide":
            shaping -= 0.02

        step_reward = round(
            yield_component
            + health_component
            + sustainability
            - pest_damage
            + shaping, 4
        )

        # Log daily averages
        self.log.daily_health.append(round(avg_health, 4))
        self.log.daily_pest_risk.append(round(avg_pest, 4))

        return Reward(
            step_reward=step_reward,
            episode_reward=self.episode_reward,
            yield_component=yield_component,
            health_component=health_component,
            efficiency_component=round(sustainability, 4),
            done=False,
            info={
                "avg_health": round(avg_health, 4),
                "avg_pest":   round(avg_pest, 4),
                "shaping":    shaping,
            }
        )

    def _build_observation(self) -> Observation:
        """Constructs the full Observation object for the agent."""
        day = self.day

        # Season phase
        if day <= self.episode_days * 0.2:
            phase = "planting"
        elif day <= self.episode_days * 0.8:
            phase = "growing"
        else:
            phase = "harvest"

        # Weather
        w = self.current_weather
        weather_obs = WeatherObservation(
            event=w.event.value,
            rainfall_mm=w.rainfall_mm,
            temperature=w.temperature,
            humidity=w.humidity,
            frost=w.frost,
            rain_probability=w.rain_probability,
        )

        # Zones
        zone_obs = [
            ZoneObservation(**z)
            for z in self.grid.all_zones_dict()
        ]

        # Market
        market_prices = self.market.get_current_prices()

        return Observation(
            day=day,
            season_phase=phase,
            budget_left=round(self.budget, 4),
            market_prices=market_prices,
            weather=weather_obs,
            zones=zone_obs,
            history=self.history[-5:],
        )

    def _estimate_max_revenue(self) -> float:
        """
        Estimates the theoretical maximum revenue for grading.
        Assumes perfect yield (1.0) for all zones at base price.
        """
        crops   = self.config["crops"]
        n_zones = self.config["num_zones"]
        avg_base = sum(
            __import__("env.market", fromlist=["BASE_PRICES"])
            .BASE_PRICES.get(c, 3.0)
            for c in crops
        ) / len(crops)
        return round(avg_base * n_zones * 100, 2)

    def _finalize_log(self) -> None:
        """
        Finalizes episode log when done=True.
        Counts failed zones, checks pest containment.
        """
        zones = list(self.grid.zones.values())
        self.log.budget_end   = round(self.budget, 4)
        self.log.zones_failed = sum(
            1 for z in zones if z.crop_health < 0.1
        )
        # Pest contained = no zone has pest_risk > 0.8 at episode end
        self.log.pest_contained = all(
            z.pest_risk <= 0.8 for z in zones
        )

    def grade(self) -> dict:
        """Grades completed episode. Ensures score is strictly in (0, 1)."""
        if not self.done:
            return {"error": "Episode not finished yet. Keep stepping."}
        grader = GraderFactory.get(self.task_id)
        result = grader.grade(self.log)
        # Enforce strict bounds required by Phase 2 validator
        result["final_score"] = max(0.01, min(0.99, result["final_score"]))
        return result

if __name__ == "__main__":
    import json

    print("=" * 50)
    print("AgriculturalEnvironment Test")
    print("=" * 50)

    # Test 1 — reset() returns valid Observation
    print("\n--- Test 1: reset() ---")
    env = AgriculturalEnvironment("task_1_easy")
    obs = env.reset()
    print(f"Day:          {obs.day}")
    print(f"Season phase: {obs.season_phase}")
    print(f"Budget left:  {obs.budget_left}")
    print(f"Zones:        {len(obs.zones)}")
    print(f"Weather:      {obs.weather.event}")
    assert obs.day == 1,              "Day should be 1 after reset"
    assert len(obs.zones) == 2,       "Task 1 should have 2 zones"
    assert obs.budget_left == 500.0,  "Task 1 budget should be 500"
    print("SUCCESS: reset() passed")

    # Test 2 — step() with irrigate action
    print("\n--- Test 2: step() irrigate ---")
    action = Action(action_type="irrigate", zone=0, amount=10.0)
    obs2, reward2, done2, info2 = env.step(action)
    print(f"Day after step:  {obs2.day}")
    print(f"Step reward:     {reward2.step_reward}")
    print(f"Budget left:     {obs2.budget_left}")
    print(f"Done:            {done2}")
    assert obs2.day == 2,            "Day should advance to 2"
    assert obs2.budget_left < 500.0, "Budget should decrease after irrigate"
    assert not done2,                "Should not be done on day 2"
    print("SUCCESS: step() irrigate passed")

    # Test 3 — step() with wait action
    print("\n--- Test 3: step() wait ---")
    action_wait = Action(action_type="wait", days=1)
    obs3, reward3, done3, info3 = env.step(action_wait)
    print(f"Step reward: {reward3.step_reward}")
    assert obs3.day == 3, "Day should advance to 3"
    print("SUCCESS: step() wait passed")

    # Test 4 — state() returns full state
    print("\n--- Test 4: state() ---")
    s = env.state()
    print(json.dumps({
        k: v for k, v in s.items()
        if k not in ("zones", "history")
    }, indent=2))
    assert "day"            in s
    assert "budget"         in s
    assert "zones"          in s
    assert "weather"        in s
    assert "market"         in s
    assert "episode_reward" in s
    print("SUCCESS: state() passed")

    # Test 5 — Full episode Task 1 (run to completion)
    print("\n--- Test 5: Full Task 1 episode ---")
    env2  = AgriculturalEnvironment("task_1_easy")
    obs   = env2.reset()
    done  = False
    steps = 0
    while not done:
        # Simple policy: irrigate zone 0 every 3 days, else wait
        if steps % 3 == 0:
            act = Action(action_type="irrigate", zone=0, amount=10.0)
        else:
            act = Action(action_type="wait", days=1)
        obs, reward, done, info = env2.step(act)
        steps += 1
    print(f"Episode finished on day: {obs.day}")
    print(f"Total steps:             {steps}")
    print(f"Final episode reward:    {reward.episode_reward}")
    assert done,         "done must be True at end of episode"
    assert steps <= 30,  "Task 1 should end within 30 steps"
    print("SUCCESS: Full Task 1 episode passed")

    # Test 6 — grade() after episode
    print("\n--- Test 6: grade() ---")
    result = env2.grade()
    print(json.dumps(result, indent=2))
    assert "final_score"  in result
    assert 0.0 < result["final_score"] < 1.0  # validator: strictly between 0 and 1 (exclusive)
    print("SUCCESS: grade() passed")

    # Sanity: run episode with harvest action and verify score improves
    print("\n--- Test 6b: Sanity harvest check ---")
    env_check = AgriculturalEnvironment("task_1_easy")
    env_check.episode_days = 90  # Wheat takes 61 days to mature, need enough days
    obs = env_check.reset()
    env_check.episode_days = 90  # reset() might override it? Let's trace it.
    done = False
    steps = 0
    while not done:
        zones = obs.zones
        # Irrigate early, harvest when ready
        action_taken = False
        for i, z in enumerate(zones):
            if z.growth_stage == 4 and not z.is_harvested:
                act = Action(action_type="harvest", zone=i)
                action_taken = True
                break
        if not action_taken:
            if steps % 2 == 0:
                act = Action(action_type="irrigate", zone=0, amount=10.0)
            else:
                act = Action(action_type="wait", days=1)
        obs, reward, done, info = env_check.step(act)
        steps += 1

    result_check = env_check.grade()
    print(f"Sanity harvest check score: {result_check['final_score']}")
    print(f"Harvest count: {result_check['breakdown']['harvest_count']}")
    print(f"Total revenue: {result_check['breakdown']['total_revenue']}")
    assert result_check['breakdown']['harvest_count'] > 0, \
        "Harvest policy should produce at least 1 harvest"
    print("SUCCESS: Harvest sanity check passed")

    # Test 7 — reset() after done restores state
    print("\n--- Test 7: reset() after done ---")
    obs_fresh = env2.reset()
    assert obs_fresh.day == 1,           "Day must reset to 1"
    assert obs_fresh.budget_left == 500.0,"Budget must reset to 500"
    assert not env2.done,                "done must be False after reset"
    print("SUCCESS: reset() after done passed")

    # Test 8 — Task 3 initializes correctly
    print("\n--- Test 8: Task 3 initialization ---")
    env3 = AgriculturalEnvironment("task_3_hard")
    obs3 = env3.reset()
    assert len(obs3.zones) == 16,         "Task 3 needs 16 zones"
    assert obs3.budget_left == 1000.0,    "Task 3 budget is 1000"
    print(f"Task 3 zones:  {len(obs3.zones)}")
    print(f"Task 3 budget: {obs3.budget_left}")
    print("SUCCESS: Task 3 initialization passed")

    # Test 9 — RuntimeError when stepping after done
    print("\n--- Test 9: Step after done raises error ---")
    env4 = AgriculturalEnvironment("task_1_easy")
    env4.reset()
    env4.done = True
    try:
        env4.step(Action(action_type="wait"))
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        print(f"Correctly raised: {e}")
    print("SUCCESS: RuntimeError on done passed")

    print("\nSUCCESS: environment.py test complete")
