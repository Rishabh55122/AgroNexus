from dataclasses import dataclass, field
from typing import Optional
from env.farm import FarmZone, GrowthStage
from env.grid import FarmGrid
from env.market import MarketEngine, BASE_PRICES

@dataclass
class EpisodeLog:
    """
    Complete record of one episode.
    Populated by environment.py during step() calls.
    Passed to graders at episode end.
    """
    task_id:           str
    total_days:        int          = 0
    budget_start:      float        = 1000.0
    budget_end:        float        = 0.0
    water_used:        float        = 0.0
    water_budget:      float        = 500.0
    actions_taken:     list[dict]   = field(default_factory=list)
    harvest_events:    list[dict]   = field(default_factory=list)
    sell_events:       list[dict]   = field(default_factory=list)
    pest_events:       list[dict]   = field(default_factory=list)
    daily_health:      list[float]  = field(default_factory=list)
    daily_pest_risk:   list[float]  = field(default_factory=list)
    final_zones:       list[dict]   = field(default_factory=list)
    total_revenue:     float        = 0.0
    max_possible_revenue: float     = 0.0
    zones_failed:      int          = 0
    zones_total:       int          = 0
    pest_contained:    bool         = False
    scenario:          str          = "normal"

class BaseGrader:
    """
    Base class for all task graders.
    All graders must return scores in [0.0, 1.0].
    All graders must be deterministic given the same EpisodeLog.
    """

    def grade(self, log: EpisodeLog) -> dict:
        """
        Grade the episode. Returns full scoring breakdown.
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def _clamp(self, value: float) -> float:
        """Ensures score is strictly within [0.0, 1.0]."""
        return round(max(0.0, min(1.0, value)), 4)

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Division that returns 0.0 instead of ZeroDivisionError."""
        if denominator == 0:
            return 0.0
        return numerator / denominator

class Task1Grader(BaseGrader):
    """
    Grader for Task 1: Normal season, 2 zones, stable weather.
    Focus: Did the agent irrigate correctly and harvest at the right time?

    Scoring breakdown:
        yield_score      (0.4) — actual yield vs maximum possible
        timing_score     (0.3) — harvested at READY stage (not early/late)
        efficiency_score (0.3) — water used vs theoretical minimum
    """

    def grade(self, log: EpisodeLog) -> dict:
        yield_score      = self._grade_yield(log)
        timing_score     = self._grade_timing(log)
        efficiency_score = self._grade_efficiency(log)

        final_score = (
            yield_score      * 0.40 +
            timing_score     * 0.30 +
            efficiency_score * 0.30
        )

        return {
            "task_id":        "task_1_easy",
            "final_score":    self._clamp(final_score),
            "yield_score":    self._clamp(yield_score),
            "timing_score":   self._clamp(timing_score),
            "efficiency_score": self._clamp(efficiency_score),
            "breakdown": {
                "total_revenue":    log.total_revenue,
                "water_used":       log.water_used,
                "harvest_count":    len(log.harvest_events),
                "zones_failed":     log.zones_failed,
            }
        }

    def _grade_yield(self, log: EpisodeLog) -> float:
        """Ratio of actual revenue to maximum possible revenue."""
        return self._safe_divide(
            log.total_revenue,
            log.max_possible_revenue
        )

    def _grade_timing(self, log: EpisodeLog) -> float:
        """
        Score based on how many harvests happened at READY stage.
        Penalizes early harvest (stage < 4) and missed harvests.
        """
        if not log.harvest_events:
            return 0.0
        correct = sum(
            1 for h in log.harvest_events
            if h.get("growth_stage") == int(GrowthStage.READY)
        )
        return self._safe_divide(correct, len(log.harvest_events))

    def _grade_efficiency(self, log: EpisodeLog) -> float:
        """
        Penalizes water overuse.
        Perfect score = used exactly what crops needed.
        Score degrades linearly as overuse increases.
        """
        if log.water_used == 0:
            return 0.0
        ratio = self._safe_divide(log.water_budget, log.water_used)
        return self._clamp(ratio)

class Task2Grader(BaseGrader):
    """
    Grader for Task 2: Drought mid-season, limited water supply.
    Focus: Did the agent prioritize zones correctly under constraint?

    Scoring breakdown:
        survival_score   (0.35) — zones kept alive vs total zones
        water_score      (0.30) — stayed within water budget
        yield_score      (0.25) — revenue under drought conditions
        pest_score       (0.10) — managed pest risk during drought
    """

    def grade(self, log: EpisodeLog) -> dict:
        survival_score = self._grade_survival(log)
        water_score    = self._grade_water(log)
        yield_score    = self._grade_yield(log)
        pest_score     = self._grade_pest(log)

        final_score = (
            survival_score * 0.35 +
            water_score    * 0.30 +
            yield_score    * 0.25 +
            pest_score     * 0.10
        )

        return {
            "task_id":       "task_2_medium",
            "final_score":   self._clamp(final_score),
            "survival_score": self._clamp(survival_score),
            "water_score":   self._clamp(water_score),
            "yield_score":   self._clamp(yield_score),
            "pest_score":    self._clamp(pest_score),
            "breakdown": {
                "zones_survived": log.zones_total - log.zones_failed,
                "zones_failed":   log.zones_failed,
                "water_used":     log.water_used,
                "water_budget":   log.water_budget,
                "total_revenue":  log.total_revenue,
            }
        }

    def _grade_survival(self, log: EpisodeLog) -> float:
        """Fraction of zones that survived (health > 0.1) at episode end."""
        survived = log.zones_total - log.zones_failed
        return self._safe_divide(survived, log.zones_total)

    def _grade_water(self, log: EpisodeLog) -> float:
        """
        Full score if water used <= water budget.
        Linear penalty for going over budget.
        Zero score if water used >= 2x budget.
        """
        if log.water_used <= log.water_budget:
            return 1.0
        overuse = log.water_used - log.water_budget
        penalty = self._safe_divide(overuse, log.water_budget)
        return self._clamp(1.0 - penalty)

    def _grade_yield(self, log: EpisodeLog) -> float:
        """Revenue ratio — adjusted for drought difficulty."""
        base = self._safe_divide(
            log.total_revenue,
            log.max_possible_revenue
        )
        # Drought makes 40% of max a good result — scale accordingly
        return self._clamp(base * 1.5)

    def _grade_pest(self, log: EpisodeLog) -> float:
        """
        Score based on average pest risk across episode.
        Low average pest risk = high score.
        """
        if not log.daily_pest_risk:
            return 1.0
        avg_risk = sum(log.daily_pest_risk) / len(log.daily_pest_risk)
        return self._clamp(1.0 - avg_risk)

class Task3Grader(BaseGrader):
    """
    Grader for Task 3: Drought + pest outbreak + market swings + budget crisis.
    Focus: Multi-objective optimization under cascading failures.

    Composite scoring:
        profit_score       (0.50) — net profit vs max possible
        pest_control_score (0.30) — pest contained before spreading
        sustainability_score (0.20) — budget managed, no overuse
    """

    def grade(self, log: EpisodeLog) -> dict:
        profit_score       = self._grade_profit(log)
        pest_control_score = self._grade_pest_control(log)
        sustainability     = self._grade_sustainability(log)

        final_score = (
            profit_score       * 0.50 +
            pest_control_score * 0.30 +
            sustainability     * 0.20
        )

        return {
            "task_id":             "task_3_hard",
            "final_score":         self._clamp(final_score),
            "profit_score":        self._clamp(profit_score),
            "pest_control_score":  self._clamp(pest_control_score),
            "sustainability_score": self._clamp(sustainability),
            "breakdown": {
                "total_revenue":       log.total_revenue,
                "max_possible_revenue":log.max_possible_revenue,
                "pest_contained":      log.pest_contained,
                "budget_end":          log.budget_end,
                "zones_failed":        log.zones_failed,
                "sell_events":         len(log.sell_events),
            }
        }

    def _grade_profit(self, log: EpisodeLog) -> float:
        """
        Net profit score.
        Accounts for market crash — 25% of max is excellent in crisis.
        """
        base = self._safe_divide(
            log.total_revenue,
            log.max_possible_revenue
        )
        # Scale up — crisis makes 25% of max a strong result
        return self._clamp(base * 2.0)

    def _grade_pest_control(self, log: EpisodeLog) -> float:
        """
        Rewards containing pest outbreak before it spreads grid-wide.
        Full score if pest_contained = True.
        Partial score based on how few zones were affected.
        """
        if log.pest_contained:
            return 1.0
        if not log.pest_events:
            return 1.0
        # Score degrades with each pest event (spread event)
        spread_events = len(log.pest_events)
        penalty = min(1.0, spread_events * 0.05)
        return self._clamp(1.0 - penalty)

    def _grade_sustainability(self, log: EpisodeLog) -> float:
        """
        Composite sustainability:
            0.5 — budget not exhausted
            0.5 — water not overused
        """
        # Budget score
        if log.budget_end > 0:
            budget_score = 1.0
        else:
            budget_score = 0.0

        # Water score
        water_score = self._clamp(
            self._safe_divide(log.water_budget, max(log.water_used, 1))
        )

        return (budget_score * 0.5) + (water_score * 0.5)

class GraderFactory:
    """
    Returns the correct grader for a given task ID.
    Used by the API /grader endpoint.
    """

    _graders = {
        "task_1_easy":   Task1Grader,
        "task_2_medium": Task2Grader,
        "task_3_hard":   Task3Grader,
    }

    @classmethod
    def get(cls, task_id: str) -> BaseGrader:
        """Returns grader instance for given task_id."""
        if task_id not in cls._graders:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid: {list(cls._graders.keys())}"
            )
        return cls._graders[task_id]()

    @classmethod
    def list_tasks(cls) -> list[str]:
        """Returns all registered task IDs."""
        return list(cls._graders.keys())

if __name__ == "__main__":
    import json

    print("=" * 50)
    print("Graders Test")
    print("=" * 50)

    # Helper to build a mock EpisodeLog
    def make_log(task_id, **kwargs) -> EpisodeLog:
        defaults = dict(
            task_id=task_id,
            total_days=30,
            budget_start=1000.0,
            budget_end=200.0,
            water_used=300.0,
            water_budget=500.0,
            total_revenue=450.0,
            max_possible_revenue=600.0,
            zones_failed=0,
            zones_total=2,
            pest_contained=True,
            scenario="normal",
            daily_health=[0.9] * 30,
            daily_pest_risk=[0.1] * 30,
            harvest_events=[
                {"zone_id": "A1", "growth_stage": 4, "yield": 0.85},
                {"zone_id": "A2", "growth_stage": 4, "yield": 0.90},
            ],
            sell_events=[
                {"crop": "wheat", "amount": 100, "revenue": 280.0},
            ],
            pest_events=[],
            final_zones=[],
            actions_taken=[],
        )
        defaults.update(kwargs)
        return EpisodeLog(**defaults)

    # Test 1 — Task1Grader perfect episode
    print("\n--- Test 1: Task1Grader perfect episode ---")
    log1 = make_log(
        "task_1_easy",
        total_revenue=600.0,
        max_possible_revenue=600.0,
        water_used=300.0,
        water_budget=500.0,
    )
    result1 = Task1Grader().grade(log1)
    print(json.dumps(result1, indent=2))
    assert result1["final_score"] >= 0.7, \
        "Perfect episode should score >= 0.7"
    assert 0.0 <= result1["final_score"] <= 1.0, \
        "Score must be in [0, 1]"
    print("SUCCESS: Task1Grader perfect episode passed")

    # Test 2 — Task1Grader poor episode
    print("\n--- Test 2: Task1Grader poor episode ---")
    log2 = make_log(
        "task_1_easy",
        total_revenue=100.0,
        max_possible_revenue=600.0,
        water_used=900.0,
        water_budget=500.0,
        harvest_events=[
            {"zone_id": "A1", "growth_stage": 2, "yield": 0.2},
        ],
    )
    result2 = Task1Grader().grade(log2)
    print(json.dumps(result2, indent=2))
    assert result2["final_score"] < result1["final_score"], \
        "Poor episode must score lower than perfect episode"
    print("SUCCESS: Task1Grader poor episode passed")

    # Test 3 — Task2Grader water budget exceeded
    print("\n--- Test 3: Task2Grader water exceeded ---")
    log3 = make_log(
        "task_2_medium",
        water_used=1100.0,
        water_budget=500.0,
        zones_total=4,
        zones_failed=2,
        daily_pest_risk=[0.4] * 60,
    )
    result3 = Task2Grader().grade(log3)
    print(json.dumps(result3, indent=2))
    assert result3["water_score"] < 0.5, \
        "Exceeding water budget should give low water_score"
    assert 0.0 <= result3["final_score"] <= 1.0
    print("SUCCESS: Task2Grader water exceeded passed")

    # Test 4 — Task3Grader crisis scenario
    print("\n--- Test 4: Task3Grader crisis scenario ---")
    log4 = make_log(
        "task_3_hard",
        scenario="crisis",
        total_revenue=200.0,
        max_possible_revenue=1200.0,
        pest_contained=False,
        budget_end=0.0,
        zones_total=16,
        zones_failed=6,
        pest_events=[{"day": i, "zone": "A1"} for i in range(10)],
        daily_pest_risk=[0.6] * 90,
    )
    result4 = Task3Grader().grade(log4)
    print(json.dumps(result4, indent=2))
    assert result4["pest_control_score"] < 1.0, \
        "Uncontained pest should reduce pest_control_score"
    assert 0.0 <= result4["final_score"] <= 1.0
    print("SUCCESS: Task3Grader crisis scenario passed")

    # Test 5 — All scores strictly in [0.0, 1.0]
    print("\n--- Test 5: Score bounds for all graders ---")
    extreme_log = make_log(
        "task_1_easy",
        total_revenue=0.0,
        max_possible_revenue=0.0,
        water_used=0.0,
        water_budget=0.0,
        harvest_events=[],
        zones_failed=10,
        zones_total=10,
    )
    for grader_cls, tid in [
        (Task1Grader, "task_1_easy"),
        (Task2Grader, "task_2_medium"),
        (Task3Grader, "task_3_hard"),
    ]:
        extreme_log.task_id = tid
        r = grader_cls().grade(extreme_log)
        assert 0.0 <= r["final_score"] <= 1.0, \
            f"{tid} score out of bounds: {r['final_score']}"
        print(f"  {tid}: {r['final_score']} SUCCESS:")
    print("SUCCESS: Score bounds passed for all graders")

    # Test 6 — GraderFactory returns correct grader
    print("\n--- Test 6: GraderFactory ---")
    for task_id in GraderFactory.list_tasks():
        grader = GraderFactory.get(task_id)
        print(f"  {task_id} -> {type(grader).__name__}")
        assert isinstance(grader, BaseGrader)
    try:
        GraderFactory.get("invalid_task")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Invalid task correctly rejected: {e}")
    print("SUCCESS: GraderFactory passed")

    # Test 7 — Higher revenue = higher score (monotonic)
    print("\n--- Test 7: Monotonic scoring ---")
    scores = []
    for revenue in [0, 150, 300, 450, 600]:
        log = make_log(
            "task_1_easy",
            total_revenue=float(revenue),
            max_possible_revenue=600.0,
        )
        r = Task1Grader().grade(log)
        scores.append(r["final_score"])
        print(f"  Revenue ${revenue} -> score {r['final_score']}")
    assert scores == sorted(scores), \
        "Higher revenue must produce higher or equal score"
    print("SUCCESS: Monotonic scoring passed")

    print("\nSUCCESS: graders.py test complete")
