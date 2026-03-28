import random
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class CropMarket(str, Enum):
    LOCAL  = "local"   # lower price, always available
    EXPORT = "export"  # higher price, only available certain days

BASE_PRICES = {
    "wheat": 2.80,
    "corn":  3.20,
    "soy":   4.50,
}

MARKET_VOLATILITY = {
    "wheat": 0.08,   # ±8% daily swing
    "corn":  0.10,   # ±10% daily swing
    "soy":   0.15,   # ±15% daily swing
}

@dataclass
class PriceSnapshot:
    day:          int
    crop:         str
    local_price:  float
    export_price: float
    export_open:  bool    # export market available today?
    trend:        str     # "rising" | "falling" | "stable"

    def best_price(self) -> float:
        """Returns highest available price today."""
        if self.export_open:
            return max(self.local_price, self.export_price)
        return self.local_price

    def to_dict(self) -> dict:
        return {
            "day":          self.day,
            "crop":         self.crop,
            "local_price":  round(self.local_price, 4),
            "export_price": round(self.export_price, 4),
            "export_open":  self.export_open,
            "trend":        self.trend,
            "best_price":   round(self.best_price(), 4),
        }

class MarketEngine:
    """
    Simulates dynamic crop market prices for the Precision Agriculture env.

    Key behaviors:
    - Prices fluctuate daily using a mean-reverting random walk
    - Export market opens only on certain days (adds strategic timing)
    - Crisis scenario applies a market crash after day 45
    - Prices never drop below 50% or rise above 200% of base price
    - Fixed seed ensures reproducible grader scoring
    """

    def __init__(
        self,
        scenario:    str = "normal",
        seed:        Optional[int] = 42,
        crops:       list[str] = None,
    ):
        self.scenario  = scenario
        self.seed      = seed
        self.rng       = random.Random(seed)
        self.day       = 0
        self.crops     = crops or ["wheat", "corn", "soy"]

        # Current prices start at base
        self.prices: dict[str, float] = {
            crop: BASE_PRICES[crop] for crop in self.crops
        }

        # Price history per crop
        self.history: dict[str, list[PriceSnapshot]] = {
            crop: [] for crop in self.crops
        }

        # Export opens every 7 days (weekly market day)
        self.export_interval = 7

        # Crisis crash triggers after day 45
        self.crash_active    = False
        self.crash_day       = 45 if scenario == "crisis" else None

    def step(self) -> dict[str, PriceSnapshot]:
        """
        Advances market by one day.
        Updates all crop prices using mean-reverting random walk.
        Returns dict of {crop: PriceSnapshot} for current day.
        """
        self.day += 1

        # Activate crash if day reached
        if self.crash_day and self.day >= self.crash_day:
            self.crash_active = True

        # Export market opens every 7 days
        export_open = (self.day % self.export_interval == 0)

        snapshots = {}
        for crop in self.crops:
            snapshot = self._update_price(crop, export_open)
            self.history[crop].append(snapshot)
            snapshots[crop] = snapshot

        return snapshots

    def _update_price(
        self,
        crop: str,
        export_open: bool,
    ) -> PriceSnapshot:
        """
        Updates price for one crop using mean-reverting random walk.

        Formula:
            change = volatility * gauss(0, 1)
            reversion = mean_reversion_rate * (base - current)
            new_price = current * (1 + change) + reversion

        Crisis crash: prices drop 30% after crash_day.
        Price bounds: [base * 0.5, base * 2.0]
        """
        base       = BASE_PRICES[crop]
        volatility = MARKET_VOLATILITY[crop]
        current    = self.prices[crop]

        # Mean reversion pulls price back toward base
        mean_reversion_rate = 0.05
        reversion  = mean_reversion_rate * (base - current)

        # Random daily change
        change     = volatility * self.rng.gauss(0, 1)

        # Apply crisis crash — prices drop 30%
        crash_penalty = -0.30 if self.crash_active else 0.0

        new_price  = current * (1 + change + crash_penalty) + reversion

        # Clamp to bounds
        new_price  = max(base * 0.5, min(base * 2.0, new_price))
        self.prices[crop] = round(new_price, 4)

        # Determine trend from last snapshot
        trend = "stable"
        if self.history[crop]:
            last = self.history[crop][-1].local_price
            if new_price > last * 1.01:
                trend = "rising"
            elif new_price < last * 0.99:
                trend = "falling"

        # Export price is 20% premium over local
        export_price = round(new_price * 1.20, 4)

        return PriceSnapshot(
            day=self.day,
            crop=crop,
            local_price=round(new_price, 4),
            export_price=export_price,
            export_open=export_open,
            trend=trend,
        )

    def get_current_prices(self) -> dict:
        """
        Returns current best price for all crops.
        Used to build the observation space.
        """
        result = {}
        for crop in self.crops:
            export_open = (self.day % self.export_interval == 0)
            local       = self.prices[crop]
            export      = round(local * 1.20, 4)
            result[crop] = {
                "local_price":  round(local, 4),
                "export_price": export,
                "export_open":  export_open,
                "best_price":   export if export_open else round(local, 4),
            }
        return result

    def calculate_revenue(
        self,
        crop:   str,
        amount: float,
        market: str = "local",
    ) -> float:
        """
        Calculates revenue for selling a given amount of crop.

        Args:
            crop:   crop type string e.g. "wheat"
            amount: tons to sell
            market: "local" or "export"

        Returns:
            Revenue in dollars. Returns 0.0 if export
            market is closed or crop not recognized.
        """
        if crop not in self.prices:
            return 0.0

        export_open = (self.day % self.export_interval == 0)

        if market == "export" and not export_open:
            return 0.0   # export market closed today

        price = self.prices[crop]
        if market == "export":
            price = round(price * 1.20, 4)

        return round(price * amount, 4)

    def reset(self) -> None:
        """
        Resets market to day 0 with same seed.
        Produces identical price sequence — required for reproducible grading.
        """
        self.rng          = random.Random(self.seed)
        self.day          = 0
        self.crash_active = False
        self.prices       = {
            crop: BASE_PRICES[crop] for crop in self.crops
        }
        self.history      = {crop: [] for crop in self.crops}

    def summary(self) -> dict:
        """Returns season-to-date market statistics."""
        result = {}
        for crop in self.crops:
            snaps = self.history[crop]
            if not snaps:
                continue
            prices = [s.local_price for s in snaps]
            result[crop] = {
                "base_price":    BASE_PRICES[crop],
                "current_price": round(self.prices[crop], 4),
                "min_price":     round(min(prices), 4),
                "max_price":     round(max(prices), 4),
                "avg_price":     round(sum(prices) / len(prices), 4),
                "crash_active":  self.crash_active,
                "export_days":   sum(
                    1 for s in snaps if s.export_open
                ),
            }
        return result

if __name__ == "__main__":
    import json

    print("=" * 50)
    print("MarketEngine Test")
    print("=" * 50)

    # Test 1 — Normal scenario, 30 days
    print("\n--- Test 1: Normal scenario, 30 days ---")
    market = MarketEngine(scenario="normal", seed=42)
    for _ in range(30):
        market.step()
    summary = market.summary()
    print(json.dumps(summary, indent=2))
    for crop in ["wheat", "corn", "soy"]:
        base = BASE_PRICES[crop]
        assert summary[crop]["min_price"] >= base * 0.5, \
            f"{crop} price dropped below 50% floor"
        assert summary[crop]["max_price"] <= base * 2.0, \
            f"{crop} price exceeded 200% ceiling"
    print("SUCCESS: 1. Price bounds respected")

    # Test 2 — Export market opens every 7 days
    print("\n--- Test 2: Export market interval ---")
    market2 = MarketEngine(scenario="normal", seed=42)
    export_days = []
    for day in range(1, 31):
        snaps = market2.step()
        if snaps["wheat"].export_open:
            export_days.append(day)
    print(f"Export open days: {export_days}")
    assert export_days == [7, 14, 21, 28], \
        f"Export should open on days 7,14,21,28 — got {export_days}"
    print("SUCCESS: 2. Export interval passed")

    # Test 3 — Crisis crash drops prices after day 45
    print("\n--- Test 3: Crisis market crash ---")
    market3 = MarketEngine(scenario="crisis", seed=42)
    prices_before_crash = {}
    prices_after_crash  = {}
    for day in range(1, 91):
        snaps = market3.step()
        if day == 44:
            prices_before_crash = {
                c: snaps[c].local_price for c in ["wheat","corn","soy"]
            }
        if day == 50:
            prices_after_crash = {
                c: snaps[c].local_price for c in ["wheat","corn","soy"]
            }
    print(f"Prices day 44 (before): {prices_before_crash}")
    print(f"Prices day 50 (after):  {prices_after_crash}")
    for crop in ["wheat", "corn", "soy"]:
        assert prices_after_crash[crop] < prices_before_crash[crop], \
            f"{crop} price should drop after crash"
    print("SUCCESS: 3. Crisis crash passed")

    # Test 4 — Reproducibility
    print("\n--- Test 4: Reproducibility ---")
    m1 = MarketEngine(scenario="normal", seed=7)
    m2 = MarketEngine(scenario="normal", seed=7)
    seq1 = [m1.step()["wheat"].local_price for _ in range(10)]
    seq2 = [m2.step()["wheat"].local_price for _ in range(10)]
    print(f"Sequence 1: {seq1}")
    print(f"Sequence 2: {seq2}")
    assert seq1 == seq2, "Same seed must produce same sequence"
    print("SUCCESS: 4. Reproducibility passed")

    # Test 5 — Reset restores sequence
    print("\n--- Test 5: Reset test ---")
    m3 = MarketEngine(scenario="normal", seed=99)
    seq_before = [m3.step()["corn"].local_price for _ in range(5)]
    m3.reset()
    seq_after  = [m3.step()["corn"].local_price for _ in range(5)]
    print(f"Before reset: {seq_before}")
    print(f"After  reset: {seq_after}")
    assert seq_before == seq_after, "Reset must restore price sequence"
    print("SUCCESS: 5. Reset passed")

    # Test 6 — Revenue calculation
    print("\n--- Test 6: Revenue calculation ---")
    m4 = MarketEngine(scenario="normal", seed=42)
    for _ in range(7):
        m4.step()   # day 7 — export market open
    rev_local  = m4.calculate_revenue("wheat", 100.0, "local")
    rev_export = m4.calculate_revenue("wheat", 100.0, "export")
    rev_closed = m4.calculate_revenue("wheat", 100.0, "export")
    print(f"Day 7 local  revenue (100t wheat): ${rev_local}")
    print(f"Day 7 export revenue (100t wheat): ${rev_export}")
    assert rev_export > rev_local,  "Export price should exceed local"
    print("SUCCESS: 6. Revenue calculation passed")

    # Test 7 — get_current_prices structure
    print("\n--- Test 7: get_current_prices structure ---")
    m5 = MarketEngine(scenario="normal", seed=42)
    m5.step()
    prices = m5.get_current_prices()
    print(json.dumps(prices, indent=2))
    for crop in ["wheat", "corn", "soy"]:
        assert "local_price"  in prices[crop]
        assert "export_price" in prices[crop]
        assert "export_open"  in prices[crop]
        assert "best_price"   in prices[crop]
    print("SUCCESS: 7. get_current_prices structure passed")

    print("\nSUCCESS: market.py test complete")
