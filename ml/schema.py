"""
ml/schema.py

Single source of truth for the category lists and feature columns used
across the whole pipeline. Both the offline training script (Phase 4) and
the live Spark scoring job (Phase 5) import this file, so the zone/sensor
encoders can never silently drift out of sync between training and
inference — which is a classic way real ML pipelines quietly break.
"""

ZONES = [
    "Academic_Block_A", "Academic_Block_B", "Hostel_Block_1",
    "Hostel_Block_2", "Library", "Admin_Block", "Sports_Complex",
    "Cafeteria", "Lab_Complex", "Auditorium"
]

SENSOR_TYPES = ["power", "water", "hvac", "network"]

# Features the classifier is trained on, in the Random Forest model.
# zone_enc / sensor_type_enc are produced from ZONES / SENSOR_TYPES above
# via the saved encoders, never re-fit at inference time.
CLASSIFIER_FEATURES = [
    "zone_enc", "sensor_type_enc", "hour_of_day", "day_of_week",
    "occupancy_count", "reading_value", "zone_load_factor",
]

# Features used for the clustering (operational-regime) model.
CLUSTER_FEATURES = [
    "reading_value_z", "hour_of_day", "occupancy_count", "zone_load_factor",
]
