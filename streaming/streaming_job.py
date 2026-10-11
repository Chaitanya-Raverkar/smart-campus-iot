"""
streaming/streaming_job.py

Spark Structured Streaming job:
    Kafka (campus.sensor.readings)
      -> parse JSON
      -> score each micro-batch against the trained Random Forest model
      -> publish flagged anomalies to Kafka (campus.sensor.anomalies)
      -> also print them to console for live visibility while developing

DESIGN NOTE — why foreachBatch + toPandas() instead of a pandas_udf:
    pandas_udf is the "textbook" way to run a vectorized Python function
    inside Spark, but it requires Arrow serialization, which has a known
    incompatibility with newer JDKs (Java 21+ blocks the reflective memory
    access Arrow's Java library depends on, raising
    `sun.misc.Unsafe ... not available`). foreachBatch + toPandas() uses a
    plain (non-Arrow) serialization path, so this job runs regardless of
    which JDK you have installed. It also maps more naturally onto "take
    this micro-batch, run it through my sklearn model" than forcing the
    model into a per-row/vectorized UDF would.

Run (from the project root, with the venv active):
    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 streaming/streaming_job.py

First run will download the Kafka connector jar via Maven (needs internet,
one-time, then cached) — this is normal and can take a minute.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from kafka import KafkaProducer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

KAFKA_BOOTSTRAP = "localhost:9092"
INPUT_TOPIC = "campus.sensor.readings"
OUTPUT_TOPIC = "campus.sensor.anomalies"

# ml/models/ sits one level up from streaming/ — resolved from this file's
# location so the job runs correctly regardless of your current directory
MODELS_DIR = Path(__file__).resolve().parent.parent / "ml" / "models"

FEATURE_COLS = [
    "zone_enc", "sensor_type_enc", "hour_of_day", "day_of_week",
    "occupancy_count", "reading_value", "zone_load_factor",
]

# Must match the JSON shape sensor_producer.py actually emits
READING_SCHEMA = StructType([
    StructField("event_time", StringType()),
    StructField("zone", StringType()),
    StructField("sensor_type", StringType()),
    StructField("hour_of_day", DoubleType()),
    StructField("day_of_week", IntegerType()),
    StructField("occupancy_count", IntegerType()),
    StructField("reading_value", DoubleType()),
    StructField("zone_load_factor", DoubleType()),
    StructField("_simulated_is_anomaly", IntegerType()),
    StructField("_simulated_fault_kind", StringType()),
])

print(f"Loading models from {MODELS_DIR} ...")
rf_model = joblib.load(MODELS_DIR / "rf_classifier.joblib")
zone_encoder = joblib.load(MODELS_DIR / "zone_encoder.joblib")
sensor_encoder = joblib.load(MODELS_DIR / "sensor_type_encoder.joblib")
print("Models loaded.")

anomaly_producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BOOTSTRAP],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


def score_and_publish(batch_df, batch_id: int):
    """Called once per micro-batch by Spark. batch_df is a plain (static)
    Spark DataFrame containing just this batch's rows."""
    if batch_df.rdd.isEmpty():
        return

    pdf = batch_df.toPandas()

    pdf["zone_enc"] = zone_encoder.transform(pdf["zone"])
    pdf["sensor_type_enc"] = sensor_encoder.transform(pdf["sensor_type"])
    X = pdf[FEATURE_COLS]

    pdf["predicted_anomaly"] = rf_model.predict(X)
    pdf["anomaly_probability"] = rf_model.predict_proba(X)[:, 1]

    anomalies = pdf[pdf["predicted_anomaly"] == 1]
    print(f"[batch {batch_id}] scored {len(pdf)} readings, flagged {len(anomalies)} anomalies")

    for _, row in anomalies.iterrows():
        ground_truth = row.get("_simulated_is_anomaly")
        event = {
            "event_time": row["event_time"],
            "zone": row["zone"],
            "sensor_type": row["sensor_type"],
            "reading_value": float(row["reading_value"]),
            "anomaly_probability": round(float(row["anomaly_probability"]), 4),
            # ground truth from the simulator, kept ONLY for you to later
            # measure real-world precision/recall of the live model —
            # a real deployment wouldn't have this field at all
            "ground_truth_was_anomaly": (
                int(ground_truth) if pd.notna(ground_truth) else None
            ),
        }
        anomaly_producer.send(OUTPUT_TOPIC, value=event)
        print(
            f"  -> ANOMALY  zone={event['zone']:<18} sensor={event['sensor_type']:<8} "
            f"value={event['reading_value']:.2f}  p={event['anomaly_probability']:.3f}"
        )

    anomaly_producer.flush()


def main():
    spark = (
        SparkSession.builder
        .appName("CampusAnomalyDetection")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", INPUT_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = (
        raw.selectExpr("CAST(value AS STRING) as json_str")
        .select(from_json(col("json_str"), READING_SCHEMA).alias("data"))
        .select("data.*")
    )

    query = (
        parsed.writeStream
        .foreachBatch(score_and_publish)
        .outputMode("append")
        .option("checkpointLocation", "./checkpoints/anomaly_detection")
        .start()
    )

    print(
        f"Streaming job running — consuming '{INPUT_TOPIC}', "
        f"publishing anomalies to '{OUTPUT_TOPIC}'. Ctrl+C to stop."
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()
