"""
producer/sensor_producer.py

Continuous Kafka producer simulating a live smart-campus IoT sensor network.
Publishes one JSON reading per sensor "tick" to the `campus.sensor.readings`
Kafka topic.

Reuses the same anomaly-injection logic as the offline dataset generator
from the course assignment, but is driven by wall-clock time instead of a
random hour value — so the stream behaves like a real, time-of-day-aware
sensor network when left running (e.g. power load actually peaks around
1pm, not at a random moment).

Usage:
    pip install -r requirements.txt
    python sensor_producer.py
    (Ctrl+C to stop cleanly)
"""

import json
import time
import signal
import sys
from datetime import datetime

import numpy as np
from kafka import KafkaProducer

RNG = np.random.default_rng()

BOOTSTRAP_SERVERS = ["localhost:9092"]
TOPIC = "campus.sensor.readings"

ZONES = [
    "Academic_Block_A", "Academic_Block_B", "Hostel_Block_1",
    "Hostel_Block_2", "Library", "Admin_Block", "Sports_Complex",
    "Cafeteria", "Lab_Complex", "Auditorium"
]
SENSOR_TYPES = ["power", "water", "hvac", "network"]

# Fixed per-zone load factor so each zone has a consistent "personality"
# across the whole run (same idea as the offline generator).
ZONE_FACTOR = {z: round(float(RNG.uniform(0.6, 1.6)), 3) for z in ZONES}

SECONDS_BETWEEN_TICKS = 1.0     # how often we emit a batch of readings
READINGS_PER_TICK = 5           # how many sensors report per tick
ANOMALY_PROBABILITY = 0.05      # live anomaly rate (tweak to taste)


def base_profile(sensor_type, hour, zone_factor):
    """Expected (normal) sensor value given type, fractional hour-of-day, zone load factor."""
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


def make_reading(now: datetime) -> dict:
    zone = str(RNG.choice(ZONES))
    sensor_type = str(RNG.choice(SENSOR_TYPES))
    hour = now.hour + now.minute / 60
    day_of_week = now.weekday()

    occupancy = max(0, int(RNG.normal(
        60 * ZONE_FACTOR[zone] * (1 if 8 <= hour <= 18 and day_of_week < 5 else 0.2),
        15)))

    expected = base_profile(sensor_type, hour, ZONE_FACTOR[zone])
    noise = RNG.normal(0, 0.08 * abs(expected) + 0.3)
    value = expected + noise

    is_anomaly = 0
    fault_kind = None
    if RNG.random() < ANOMALY_PROBABILITY:
        is_anomaly = 1
        fault_kind = str(RNG.choice(["spike", "drift", "stuck", "dropout"]))
        if fault_kind == "spike":
            value = expected * RNG.uniform(2.2, 4.0)
        elif fault_kind == "drift":
            value = expected + expected * RNG.uniform(0.8, 1.5)
        elif fault_kind == "stuck":
            value = expected * 0.02 if sensor_type != "hvac" else 0.0
        elif fault_kind == "dropout":
            value = 0.0 if sensor_type != "hvac" else expected - RNG.uniform(8, 15)

    return {
        "event_time": now.isoformat(),
        "zone": zone,
        "sensor_type": sensor_type,
        "hour_of_day": round(hour, 3),
        "day_of_week": day_of_week,
        "occupancy_count": occupancy,
        "reading_value": round(float(value), 3),
        "zone_load_factor": ZONE_FACTOR[zone],
        # NOTE: this label only exists because WE are the ones injecting the
        # fault in this simulator. The Phase 5 Spark job will NOT read this
        # field — it predicts is_anomaly itself from the trained model. It's
        # kept here only so you can later measure the model's real-world
        # accuracy against ground truth, same as any monitored ML system in
        # production would be evaluated via a shadow/holdout signal.
        "_simulated_is_anomaly": is_anomaly,
        "_simulated_fault_kind": fault_kind,
    }


def main():
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        linger_ms=50,
    )

    def shutdown(sig, frame):
        print("\nShutting down producer...")
        producer.flush()
        producer.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    print(f"Streaming to topic '{TOPIC}' on {BOOTSTRAP_SERVERS} — Ctrl+C to stop.")
    sent = 0
    while True:
        now = datetime.now()
        for _ in range(READINGS_PER_TICK):
            reading = make_reading(now)
            # key by zone so all of a zone's readings land in the same
            # partition, preserving per-zone ordering for the consumer
            producer.send(TOPIC, key=reading["zone"], value=reading)
            sent += 1
        producer.flush()
        print(f"[{now.strftime('%H:%M:%S')}] sent {READINGS_PER_TICK} readings (total: {sent})")
        time.sleep(SECONDS_BETWEEN_TICKS)


if __name__ == "__main__":
    main()
