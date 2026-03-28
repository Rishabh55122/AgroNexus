import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class WeatherEvent(str, Enum):
    RAINY    = "rainy"
    DRY      = "dry"
    HEATWAVE = "heatwave"
    NORMAL   = "normal"
    FROST    = "frost"

class WeatherScenario(str, Enum):
    NORMAL         = "normal"
    DROUGHT        = "drought"
    PEST_OUTBREAK  = "pest_outbreak"
    CRISIS         = "crisis"

SCENARIO_WEIGHTS = {
    WeatherScenario.NORMAL: {
        WeatherEvent.RAINY:    0.30,
        WeatherEvent.DRY:      0.30,
        WeatherEvent.NORMAL:   0.30,
        WeatherEvent.HEATWAVE: 0.08,
        WeatherEvent.FROST:    0.02,
    },
    WeatherScenario.DROUGHT: {
        WeatherEvent.RAINY:    0.05,
        WeatherEvent.DRY:      0.60,
        WeatherEvent.NORMAL:   0.15,
        WeatherEvent.HEATWAVE: 0.18,
        WeatherEvent.FROST:    0.02,
    },
    WeatherScenario.PEST_OUTBREAK: {
        WeatherEvent.RAINY:    0.35,
        WeatherEvent.DRY:      0.25,
        WeatherEvent.NORMAL:   0.30,
        WeatherEvent.HEATWAVE: 0.08,
        WeatherEvent.FROST:    0.02,
    },
    WeatherScenario.CRISIS: {
        WeatherEvent.RAINY:    0.05,
        WeatherEvent.DRY:      0.55,
        WeatherEvent.NORMAL:   0.10,
        WeatherEvent.HEATWAVE: 0.28,
        WeatherEvent.FROST:    0.02,
    },
}

WEATHER_EFFECTS = {
    WeatherEvent.RAINY: {
        "rainfall_mm":  15.0,
        "temperature":  22.0,
        "humidity":     0.85,
        "frost":        False,
    },
    WeatherEvent.DRY: {
        "rainfall_mm":  0.0,
        "temperature":  32.0,
        "humidity":     0.25,
        "frost":        False,
    },
    WeatherEvent.HEATWAVE: {
        "rainfall_mm":  0.0,
        "temperature":  42.0,
        "humidity":     0.15,
        "frost":        False,
    },
    WeatherEvent.NORMAL: {
        "rainfall_mm":  5.0,
        "temperature":  27.0,
        "humidity":     0.55,
        "frost":        False,
    },
    WeatherEvent.FROST: {
        "rainfall_mm":  0.0,
        "temperature":  2.0,
        "humidity":     0.40,
        "frost":        True,
    },
}

@dataclass
class WeatherState:
    event:          WeatherEvent
    rainfall_mm:    float
    temperature:    float
    humidity:       float
    frost:          bool
    rain_probability: float
    day:            int

    def to_dict(self) -> dict:
        """Serializes weather state for the observation space."""
        return {
            "event":            self.event.value,
            "rainfall_mm":      round(self.rainfall_mm, 2),
            "temperature":      round(self.temperature, 2),
            "humidity":         round(self.humidity, 4),
            "frost":            self.frost,
            "rain_probability": round(self.rain_probability, 4),
            "day":              self.day,
        }

class WeatherEngine:
    """
    Stochastic weather engine for the Precision Agriculture environment.
    Generates daily weather events based on scenario and day of season.
    Drought scenarios automatically intensify after day 20.
    Supports fixed seeds for reproducible grader scoring.
    """

    def __init__(
        self,
        scenario: WeatherScenario = WeatherScenario.NORMAL,
        seed: Optional[int] = 42,
    ):
        self.scenario = scenario
        self.seed = seed
        self.rng = random.Random(seed)
        self.day = 0
        self.drought_active = False
        self.drought_trigger_day = None
        self.history: list[WeatherState] = []

        # Drought triggers between day 20–40 for drought/crisis scenarios
        if scenario in (WeatherScenario.DROUGHT, WeatherScenario.CRISIS):
            self.drought_trigger_day = self.rng.randint(20, 40)

    def step(self) -> WeatherState:
        """
        Advances weather by one day.
        Returns a WeatherState for the current day.
        Drought intensifies weights after drought_trigger_day.
        """
        self.day += 1

        # Activate drought if trigger day reached
        if self.drought_trigger_day and self.day >= self.drought_trigger_day:
            self.drought_active = True

        # Get weights — intensify if drought is active mid-scenario
        weights = self._get_weights()

        # Sample weather event
        events = list(weights.keys())
        probs  = list(weights.values())
        event  = self.rng.choices(events, weights=probs, k=1)[0]

        # Build weather state from effects
        effects = WEATHER_EFFECTS[event].copy()

        # Add small gaussian noise to temperature and humidity
        effects["temperature"] += self.rng.gauss(0, 1.5)
        effects["humidity"]    = max(0.0, min(1.0,
            effects["humidity"] + self.rng.gauss(0, 0.05)
        ))

        # Rain probability = weight of RAINY event in current weights
        rain_prob = weights.get(WeatherEvent.RAINY, 0.0)

        state = WeatherState(
            event=event,
            rainfall_mm=effects["rainfall_mm"],
            temperature=round(effects["temperature"], 2),
            humidity=round(effects["humidity"], 4),
            frost=effects["frost"],
            rain_probability=rain_prob,
            day=self.day,
        )

        self.history.append(state)
        return state

    def _get_weights(self) -> dict:
        """
        Returns event probability weights for current day.
        If drought is active, overrides with intensified drought weights.
        """
        if self.drought_active:
            return SCENARIO_WEIGHTS[WeatherScenario.DROUGHT]
        return SCENARIO_WEIGHTS[self.scenario]

    def reset(self) -> None:
        """
        Resets engine to day 0 with the same seed.
        Produces identical sequence — required for reproducible grading.
        """
        self.rng = random.Random(self.seed)
        self.day = 0
        self.drought_active = False
        self.history = []

        if self.scenario in (WeatherScenario.DROUGHT, WeatherScenario.CRISIS):
            self.drought_trigger_day = self.rng.randint(20, 40)

    def get_forecast(self, days: int = 3) -> list[dict]:
        """
        Returns a probabilistic forecast for the next N days.
        Does NOT advance the engine — purely informational for the agent.
        Uses a temporary RNG copy so the real sequence is unaffected.
        """
        temp_rng = random.Random(self.rng.getstate()[1][0])
        weights  = self._get_weights()
        events   = list(weights.keys())
        probs    = list(weights.values())

        forecast = []
        for i in range(days):
            event   = temp_rng.choices(events, weights=probs, k=1)[0]
            effects = WEATHER_EFFECTS[event].copy()
            forecast.append({
                "day":   self.day + i + 1,
                "event": event.value,
                **effects,
                "rain_probability": weights.get(WeatherEvent.RAINY, 0.0),
            })
        return forecast

    def summary(self) -> dict:
        """Returns season-to-date weather statistics."""
        if not self.history:
            return {}

        events      = [w.event.value for w in self.history]
        temps       = [w.temperature for w in self.history]
        rain_days   = sum(1 for w in self.history if w.rainfall_mm > 0)
        total_rain  = sum(w.rainfall_mm for w in self.history)

        return {
            "days_elapsed":   self.day,
            "drought_active": self.drought_active,
            "total_rainfall": round(total_rain, 2),
            "rain_days":      rain_days,
            "avg_temperature":round(sum(temps) / len(temps), 2),
            "heatwave_days":  events.count("heatwave"),
            "frost_days":     events.count("frost"),
            "dry_days":       events.count("dry"),
        }


if __name__ == "__main__":
    import json

    print("=" * 50)
    print("WeatherEngine Test")
    print("=" * 50)

    # Test 1 — Normal scenario, 30 days
    print("\n--- Test 1: Normal scenario, 30 days ---")
    engine = WeatherEngine(scenario=WeatherScenario.NORMAL, seed=42)
    for _ in range(30):
        engine.step()
    print(json.dumps(engine.summary(), indent=2))

    # Test 2 — Drought scenario, 60 days, check drought activates
    print("\n--- Test 2: Drought scenario, 60 days ---")
    engine2 = WeatherEngine(scenario=WeatherScenario.DROUGHT, seed=42)
    drought_activated_day = None
    for day in range(1, 61):
        state = engine2.step()
        if engine2.drought_active and drought_activated_day is None:
            drought_activated_day = day
    print(f"Drought activated on day: {drought_activated_day}")
    print(json.dumps(engine2.summary(), indent=2))

    # Test 3 — Reproducibility (same seed = same sequence)
    print("\n--- Test 3: Reproducibility check ---")
    e1 = WeatherEngine(scenario=WeatherScenario.NORMAL, seed=99)
    e2 = WeatherEngine(scenario=WeatherScenario.NORMAL, seed=99)
    seq1 = [e1.step().event.value for _ in range(10)]
    seq2 = [e2.step().event.value for _ in range(10)]
    print(f"Sequence 1: {seq1}")
    print(f"Sequence 2: {seq2}")
    print(f"Reproducible: {seq1 == seq2}")

    # Test 4 — Reset restores sequence
    print("\n--- Test 4: Reset test ---")
    e3 = WeatherEngine(scenario=WeatherScenario.NORMAL, seed=7)
    seq_before = [e3.step().event.value for _ in range(5)]
    e3.reset()
    seq_after  = [e3.step().event.value for _ in range(5)]
    print(f"Before reset: {seq_before}")
    print(f"After  reset: {seq_after}")
    print(f"Reset works:  {seq_before == seq_after}")

    # Test 5 — Forecast does not advance engine
    print("\n--- Test 5: Forecast does not advance engine ---")
    e4 = WeatherEngine(scenario=WeatherScenario.NORMAL, seed=42)
    e4.step()
    day_before   = e4.day
    forecast     = e4.get_forecast(3)
    day_after    = e4.day
    print(f"Day before forecast: {day_before}")
    print(f"Day after  forecast: {day_after}")
    print(f"Engine unchanged:    {day_before == day_after}")
    print(f"Forecast: {json.dumps(forecast, indent=2)}")

    # Test 6 — Crisis scenario has most dry + heatwave days
    print("\n--- Test 6: Crisis scenario stress test ---")
    e5 = WeatherEngine(scenario=WeatherScenario.CRISIS, seed=42)
    for _ in range(90):
        e5.step()
    summary = e5.summary()
    print(json.dumps(summary, indent=2))
    print(f"Dry + heatwave days: {summary['dry_days'] + summary['heatwave_days']}")

    print("\nSUCCESS: weather.py test complete")
