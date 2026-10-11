"""
ml/data_generator.py

Generates a historical/batch dataset used to TRAIN the models offline.
This is the same simulator logic from the course assignment, refactored
into an importable function so the training script can call it directly
instead of shelling out to a separate script.

In a real deployment this would be replaced by a consumer that reads
accumulated historical readings back out of HBase (Phase 6) — but you need
*some* historical data to train on before HBase has anything in it, so the
synthetic generator remains the bootstrap data source. This mirrors how
real anomaly-detection systems are "cold-started": train on simulated or
historical data first, then retrain periodically on real accumulated data
once the live system has been running a while (see the "Retraining" note
at the bottom of train_models.py).
"""

import numpy as np
import pandas as pd

from schema import ZONES, SENSOR_TYPES


def base_profile(sensor_type, hour, zone_factor):
    if sensor_type == "power":
        daily = 8 + 25 * np.exp(-((hour - 13) ** 2) / 18)
        return daily * zone_factor
    if sensor_type == "water":
        daily = 3 + 6 * np.exp(-((hour - 8) ** 2) / 6) + 5 * np.exp(-((hour - 19) ** 2) / 6)
        return daily * zone_factor
    if sensor_type == "hvac":
        return 24 + 1.5 * np.sin((hour / 24) * 2 * np.pi)
    if sensor_type == "network":
        daily = 20 + 60 * np.exp(-((hour - 12) ** 2) / 20)
        return daily * zone_factor


def generate(n_readings: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    zone_factor_map = {z: rng.uniform(0.6, 1.6) for z in ZONES}

    rows = []
    for _ in range(n_readings):
        zone = rng.choice(ZONES)
        sensor_type = rng.choice(SENSOR_TYPES)
        hour = rng.integers(0, 24)
        day_of_week = rng.integers(0, 7)
        occupancy = max(0, int(rng.normal(
            60 * zone_factor_map[zone] * (1 if 8 <= hour <= 18 and day_of_week < 5 else 0.2),
            15)))

        expected = base_profile(sensor_type, hour, zone_factor_map[zone])
        noise = rng.normal(0, 0.08 * abs(expected) + 0.3)
        value = expected + noise

        is_anomaly = 0
        if rng.random() < 0.09:
            is_anomaly = 1
            fault_kind = rng.choice(["spike", "drift", "stuck", "dropout"])
            if fault_kind == "spike":
                value = expected * rng.uniform(2.2, 4.0)
            elif fault_kind == "drift":
                value = expected + expected * rng.uniform(0.8, 1.5)
            elif fault_kind == "stuck":
                value = expected * 0.02 if sensor_type != "hvac" else 0.0
            elif fault_kind == "dropout":
                value = 0.0 if sensor_type != "hvac" else expected - rng.uniform(8, 15)

        rows.append({
            "zone": zone,
            "sensor_type": sensor_type,
            "hour_of_day": hour,
            "day_of_week": day_of_week,
            "occupancy_count": occupancy,
            "reading_value": round(float(value), 3),
            "zone_load_factor": round(zone_factor_map[zone], 3),
            "is_anomaly": is_anomaly,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv("historical_campus_data.csv", index=False)
    print(df.shape, "saved to historical_campus_data.csv")
