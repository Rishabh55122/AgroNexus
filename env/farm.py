from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional

class CropType(str, Enum):
    WHEAT = "wheat"
    CORN  = "corn"
    SOY   = "soy"

class GrowthStage(IntEnum):
    SEED    = 0
    SPROUT  = 1
    GROWING = 2
    MATURE  = 3
    READY   = 4

STAGE_THRESHOLDS = {
    CropType.WHEAT: [7, 14, 30, 10],
    CropType.CORN:  [10, 20, 40, 14],
    CropType.SOY:   [8, 16, 35, 12],
}

@dataclass
class FarmZone:
    zone_id:       str
    crop_type:     CropType
    soil_moisture: float = 0.6
    crop_health:   float = 1.0
    growth_stage:  GrowthStage = GrowthStage.SEED
    pest_risk:     float = 0.0
    days_in_stage: int   = 0
    yield_potential: float = 1.0
    is_harvested:  bool  = False

    def update(self, weather: dict, action_taken: Optional[str] = None) -> dict:
        old_health = self.crop_health
        old_moisture = self.soil_moisture
        
        # Moisture dynamics
        self.soil_moisture += weather.get("rainfall_mm", 0) / 50
        if action_taken == "irrigate":
            self.soil_moisture += 0.3
        self.soil_moisture *= 0.92
        self.soil_moisture = max(0.0, min(1.0, self.soil_moisture))
        
        # Health dynamics
        if self.soil_moisture < 0.2:
            self.crop_health -= 0.05
        if self.pest_risk > 0.7 and action_taken != "pesticide":
            self.crop_health -= 0.08
        if weather.get("frost") == True:
            self.crop_health -= 0.10
        if action_taken == "fertilize":
            self.crop_health = min(1.0, self.crop_health + 0.05)
        self.crop_health = max(0.0, min(1.0, self.crop_health))
        
        # Yield tracking
        self.yield_potential = min(self.yield_potential, self.crop_health)
        
        # Growth progression
        self.days_in_stage += 1
        stage_advanced = False
        
        if self.growth_stage < GrowthStage.READY:
            threshold = STAGE_THRESHOLDS[self.crop_type][self.growth_stage]
            if self.days_in_stage >= threshold and self.crop_health > 0.3:
                self.growth_stage = GrowthStage(self.growth_stage + 1)
                self.days_in_stage = 0
                stage_advanced = True
                
        return {
            "health_delta": self.crop_health - old_health,
            "moisture_delta": self.soil_moisture - old_moisture,
            "stage_advanced": stage_advanced,
        }

    def apply_pesticide(self) -> None:
        """Reduces pest risk by 60%. Cost handled by environment."""
        self.pest_risk = max(0.0, self.pest_risk - 0.6)

    def harvest(self) -> float:
        """
        Harvests the zone. Returns actual yield (0.0 if not ready).
        Marks zone as harvested — cannot harvest twice.
        """
        if self.is_harvested:
            return 0.0
        if self.growth_stage < GrowthStage.READY:
            return 0.0
        self.is_harvested = True
        return round(self.yield_potential * self.crop_health, 4)

    def days_to_ready(self) -> int:
        """Estimates remaining days until crop is ready to harvest."""
        if self.growth_stage == GrowthStage.READY:
            return 0
        remaining = 0
        thresholds = STAGE_THRESHOLDS[self.crop_type]
        # Days left in current stage
        remaining += max(0, thresholds[self.growth_stage] - self.days_in_stage)
        # Full days for all future stages
        for stage in range(self.growth_stage + 1, len(thresholds)):
            remaining += thresholds[stage]
        return remaining

    def to_dict(self) -> dict:
        """Serializes zone state for the observation space."""
        return {
            "zone_id":        self.zone_id,
            "crop_type":      self.crop_type.value,
            "soil_moisture":  round(self.soil_moisture, 4),
            "crop_health":    round(self.crop_health, 4),
            "growth_stage":   int(self.growth_stage),
            "pest_risk":      round(self.pest_risk, 4),
            "days_to_ready":  self.days_to_ready(),
            "yield_potential":round(self.yield_potential, 4),
            "is_harvested":   self.is_harvested,
        }


if __name__ == "__main__":
    import json

    print("=" * 50)
    print("FarmZone Test — 30 day simulation")
    print("=" * 50)

    zones = [
        FarmZone("A1", CropType.WHEAT),
        FarmZone("B2", CropType.CORN),
        FarmZone("C3", CropType.SOY),
    ]

    dummy_weather = {
        "rainfall_mm": 3.0,
        "temperature": 28.0,
        "humidity": 0.6,
        "frost": False,
    }

    for day in range(1, 31):
        for zone in zones:
            action = "irrigate" if zone.soil_moisture < 0.3 else None
            zone.update(dummy_weather, action)

    print("\nFinal zone states after 30 days:\n")
    for zone in zones:
        print(json.dumps(zone.to_dict(), indent=2))

    print("\nHarvest test:")
    for zone in zones:
        result = zone.harvest()
        print(f"  {zone.zone_id} ({zone.crop_type.value}): yield={result}")

    print("\nSUCCESS: farm.py test complete")
