from dataclasses import dataclass, field
from typing import Optional
from env.farm import FarmZone, CropType, GrowthStage

GRID_ROWS = 4
GRID_COLS = 4
ZONE_IDS  = [
    f"{row}{col}"
    for row in ["A", "B", "C", "D"]
    for col in ["1", "2", "3", "4"]
]

DEFAULT_CROP_LAYOUT = {
    "A1": CropType.WHEAT, "A2": CropType.WHEAT,
    "A3": CropType.CORN,  "A4": CropType.CORN,
    "B1": CropType.WHEAT, "B2": CropType.WHEAT,
    "B3": CropType.CORN,  "B4": CropType.CORN,
    "C1": CropType.SOY,   "C2": CropType.SOY,
    "C3": CropType.WHEAT, "C4": CropType.WHEAT,
    "D1": CropType.SOY,   "D2": CropType.SOY,
    "D3": CropType.SOY,   "D4": CropType.CORN,
}

class FarmGrid:
    """
    Manages the 4×4 farm zone grid with spatial interdependency.

    Key behaviors:
    - Pest risk spreads 30% into adjacent zones each day if untreated
    - Over-irrigation (moisture > 0.85) causes 20% runoff to neighbors
    - A failed zone (health < 0.1) raises neighbor stress by +0.05/day
    - Adjacency is 4-directional (N/S/E/W) — no diagonals
    """

    def __init__(
        self,
        crop_layout: Optional[dict] = None,
        num_zones: int = 16,
    ):
        self.crop_layout = crop_layout or DEFAULT_CROP_LAYOUT
        self.num_zones   = num_zones
        self.zones: dict[str, FarmZone] = {}
        self.adjacency:  dict[str, list[str]] = {}
        self._initialize_zones()
        self._build_adjacency()

    def _initialize_zones(self) -> None:
        """Creates FarmZone objects for all zones in the grid."""
        active_ids = ZONE_IDS[:self.num_zones]
        for zone_id in active_ids:
            crop = self.crop_layout.get(zone_id, CropType.WHEAT)
            self.zones[zone_id] = FarmZone(
                zone_id=zone_id,
                crop_type=crop,
            )

    def _build_adjacency(self) -> None:
        """
        Builds 4-directional adjacency map for all active zones.
        Only zones present in self.zones are included as neighbors.
        """
        rows = ["A", "B", "C", "D"]
        cols = ["1", "2", "3", "4"]

        for zone_id in self.zones:
            row_idx = rows.index(zone_id[0])
            col_idx = cols.index(zone_id[1])
            neighbors = []

            # North, South, East, West
            candidates = [
                (row_idx - 1, col_idx),  # North
                (row_idx + 1, col_idx),  # South
                (row_idx, col_idx + 1),  # East
                (row_idx, col_idx - 1),  # West
            ]
            for r, c in candidates:
                if 0 <= r < len(rows) and 0 <= c < len(cols):
                    neighbor_id = f"{rows[r]}{cols[c]}"
                    if neighbor_id in self.zones:
                        neighbors.append(neighbor_id)

            self.adjacency[zone_id] = neighbors

    def spread_pests(self) -> dict:
        """
        Spreads pest risk from each zone to its neighbors.
        Rule: each zone contributes 30% of its pest_risk to each neighbor.
        Spread is calculated from PRE-spread values to avoid order dependency.
        Returns dict of {zone_id: pest_delta} for logging.
        """
        # Snapshot current pest risks BEFORE spreading
        pre_spread = {
            zid: zone.pest_risk
            for zid, zone in self.zones.items()
        }

        deltas = {}
        for zone_id, zone in self.zones.items():
            incoming = 0.0
            for neighbor_id in self.adjacency[zone_id]:
                incoming += pre_spread[neighbor_id] * 0.30

            new_risk = min(1.0, pre_spread[zone_id] + incoming)
            delta    = new_risk - pre_spread[zone_id]
            zone.pest_risk = round(new_risk, 4)
            deltas[zone_id] = round(delta, 4)

        return deltas

    def apply_water_runoff(self) -> dict:
        """
        Over-irrigated zones (moisture > 0.85) spill 20% moisture
        into each neighbor. Prevents infinite irrigation exploit.
        Returns dict of {zone_id: moisture_delta} for logging.
        """
        pre_moisture = {
            zid: zone.soil_moisture
            for zid, zone in self.zones.items()
        }

        deltas = {}
        for zone_id, zone in self.zones.items():
            if pre_moisture[zone_id] > 0.85:
                overflow = (pre_moisture[zone_id] - 0.85) * 0.20
                for neighbor_id in self.adjacency[zone_id]:
                    neighbor = self.zones[neighbor_id]
                    neighbor.soil_moisture = min(
                        1.0,
                        neighbor.soil_moisture + overflow
                    )
                deltas[zone_id] = round(-overflow * len(
                    self.adjacency[zone_id]
                ), 4)
            else:
                deltas[zone_id] = 0.0

        return deltas

    def apply_neighbor_stress(self) -> dict:
        """
        Failed zones (crop_health < 0.1) spread stress to neighbors.
        Each failed zone raises neighbor stress by +0.05/day.
        Returns dict of {zone_id: stress_applied} for logging.
        """
        stress_map = {}
        for zone_id, zone in self.zones.items():
            stress_received = 0.0
            for neighbor_id in self.adjacency[zone_id]:
                neighbor = self.zones[neighbor_id]
                if neighbor.crop_health < 0.1:
                    stress_received += 0.05

            if stress_received > 0:
                zone.crop_health = max(
                    0.0,
                    zone.crop_health - stress_received
                )
            stress_map[zone_id] = round(stress_received, 4)

        return stress_map

    def step(self, weather: dict, actions: dict) -> dict:
        """
        Advances the entire grid by one day.

        Args:
            weather: WeatherState.to_dict() output
            actions: dict of {zone_id: action_type_string}
                     e.g. {"A1": "irrigate", "B2": "pesticide"}
                     Zones not in dict receive action=None

        Returns:
            step_info dict with per-zone deltas and spread events
        """
        step_info = {
            "zone_updates":    {},
            "pest_spread":     {},
            "water_runoff":    {},
            "neighbor_stress": {},
        }

        # 1. Update each zone individually
        for zone_id, zone in self.zones.items():
            if zone.is_harvested:
                continue
            action = actions.get(zone_id, None)
            if action == "pesticide":
                zone.apply_pesticide()
            delta = zone.update(weather, action)
            step_info["zone_updates"][zone_id] = delta

        # 2. Spread pests across grid
        step_info["pest_spread"] = self.spread_pests()

        # 3. Apply water runoff from over-irrigated zones
        step_info["water_runoff"] = self.apply_water_runoff()

        # 4. Apply neighbor stress from failed zones
        step_info["neighbor_stress"] = self.apply_neighbor_stress()

        return step_info

    def get_zone(self, zone_id: str) -> FarmZone:
        """Returns a zone by ID. Raises KeyError if not found."""
        if zone_id not in self.zones:
            raise KeyError(f"Zone '{zone_id}' not in grid.")
        return self.zones[zone_id]

    def all_zones_dict(self) -> list[dict]:
        """
        Returns list of all zone states as dicts.
        Used to build the observation space.
        """
        return [zone.to_dict() for zone in self.zones.values()]

    def reset(self) -> None:
        """Resets all zones to initial state. Rebuilds grid from scratch."""
        self._initialize_zones()
        self._build_adjacency()

    def summary(self) -> dict:
        """Returns grid-wide statistics for logging and grading."""
        zones     = list(self.zones.values())
        harvested = [z for z in zones if z.is_harvested]
        failed    = [z for z in zones if z.crop_health < 0.1]
        at_risk   = [z for z in zones if z.pest_risk > 0.5]

        return {
            "total_zones":      len(zones),
            "harvested_zones":  len(harvested),
            "failed_zones":     len(failed),
            "at_risk_zones":    len(at_risk),
            "avg_health":       round(
                sum(z.crop_health for z in zones) / len(zones), 4
            ),
            "avg_moisture":     round(
                sum(z.soil_moisture for z in zones) / len(zones), 4
            ),
            "avg_pest_risk":    round(
                sum(z.pest_risk for z in zones) / len(zones), 4
            ),
            "total_yield":      round(
                sum(z.harvest() if z.growth_stage == GrowthStage.READY
                    and not z.is_harvested else 0.0
                    for z in zones), 4
            ),
        }

if __name__ == "__main__":
    import json
    from env.weather import WeatherEngine, WeatherScenario

    print("=" * 50)
    print("FarmGrid Test")
    print("=" * 50)

    # Test 1 — Grid initializes correctly
    print("\n--- Test 1: Grid initialization ---")
    grid = FarmGrid(num_zones=16)
    print(f"Total zones:     {len(grid.zones)}")
    print(f"Zone IDs:        {list(grid.zones.keys())}")
    print(f"A1 neighbors:    {grid.adjacency['A1']}")
    print(f"B2 neighbors:    {grid.adjacency['B2']}")
    print(f"D4 neighbors:    {grid.adjacency['D4']}")
    assert len(grid.zones) == 16,        "Should have 16 zones"
    assert grid.adjacency["A1"] == ["B1", "A2"], \
        f"A1 neighbors wrong: {grid.adjacency['A1']}"
    assert len(grid.adjacency["B2"]) == 4, \
        f"B2 should have 4 neighbors"
    print("SUCCESS: 1. Grid initialization passed")

    # Test 2 — Pest spread propagates correctly
    print("\n--- Test 2: Pest spread ---")
    grid2 = FarmGrid(num_zones=16)
    grid2.zones["A1"].pest_risk = 1.0
    deltas = grid2.spread_pests()
    print(f"A1 pest risk:    {grid2.zones['A1'].pest_risk}")
    print(f"A2 pest risk:    {grid2.zones['A2'].pest_risk}")
    print(f"B1 pest risk:    {grid2.zones['B1'].pest_risk}")
    print(f"C1 pest risk:    {grid2.zones['C1'].pest_risk}")
    assert grid2.zones["A2"].pest_risk > 0, "Pest should spread to A2"
    assert grid2.zones["B1"].pest_risk > 0, "Pest should spread to B1"
    assert grid2.zones["C1"].pest_risk == 0, "C1 too far, should be 0"
    print("SUCCESS: 2. Pest spread passed")

    # Test 3 — Water runoff from over-irrigated zone
    print("\n--- Test 3: Water runoff ---")
    grid3 = FarmGrid(num_zones=16)
    grid3.zones["B2"].soil_moisture = 0.95
    before_neighbor = grid3.zones["B3"].soil_moisture
    grid3.apply_water_runoff()
    after_neighbor = grid3.zones["B3"].soil_moisture
    print(f"B3 moisture before: {before_neighbor}")
    print(f"B3 moisture after:  {after_neighbor}")
    assert after_neighbor > before_neighbor, \
        "B3 should receive runoff from B2"
    print("SUCCESS: 3. Water runoff passed")

    # Test 4 — Neighbor stress from failed zone
    print("\n--- Test 4: Neighbor stress ---")
    grid4 = FarmGrid(num_zones=16)
    grid4.zones["B2"].crop_health = 0.05  # failed zone
    before = grid4.zones["A2"].crop_health
    grid4.apply_neighbor_stress()
    after = grid4.zones["A2"].crop_health
    print(f"A2 health before stress: {before}")
    print(f"A2 health after stress:  {after}")
    assert after < before, "A2 should receive stress from failed B2"
    print("SUCCESS: 4. Neighbor stress passed")

    # Test 5 — Full grid step with weather and actions
    print("\n--- Test 5: Full grid step ---")
    engine = WeatherEngine(scenario=WeatherScenario.NORMAL, seed=42)
    grid5  = FarmGrid(num_zones=16)
    actions = {"A1": "irrigate", "B2": "pesticide"}

    for day in range(1, 11):
        weather_state = engine.step()
        info = grid5.step(weather_state.to_dict(), actions)

    print(f"After 10 days:")
    print(json.dumps(grid5.summary(), indent=2))
    assert grid5.summary()["avg_health"] > 0.5, \
        "Avg health should be above 0.5 after 10 days"
    print("SUCCESS: 5. Full grid step passed")

    # Test 6 — Reset restores all zones
    print("\n--- Test 6: Grid reset ---")
    grid5.reset()
    summary_after_reset = grid5.summary()
    print(json.dumps(summary_after_reset, indent=2))
    assert summary_after_reset["avg_health"] == 1.0, \
        "All zones should reset to health 1.0"
    assert summary_after_reset["avg_pest_risk"] == 0.0, \
        "All pest risks should reset to 0.0"
    print("SUCCESS: 6. Grid reset passed")

    # Test 7 — Small grid (2 zones for Task 1)
    print("\n--- Test 7: Small grid (2 zones) ---")
    small_layout = {"A1": CropType.WHEAT, "A2": CropType.CORN}
    grid6 = FarmGrid(crop_layout=small_layout, num_zones=2)
    print(f"Zones:      {list(grid6.zones.keys())}")
    print(f"A1 neighbors: {grid6.adjacency['A1']}")
    assert len(grid6.zones) == 2, "Should have 2 zones"
    print("SUCCESS: 7. Small grid passed")

    print("\nSUCCESS: grid.py test complete")
