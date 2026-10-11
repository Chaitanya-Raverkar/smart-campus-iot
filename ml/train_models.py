"""
ml/train_models.py

Trains the two models from your assignment on historical data and
SAVES them to ml/models/, so Phase 5's Spark Streaming job can load
already-trained models and score live readings instantly, instead of
retraining on every batch (which would be both wrong and far too slow).

Outputs written to ml/models/:
    rf_classifier.joblib        - trained Random Forest anomaly classifier
    zone_encoder.joblib         - fitted LabelEncoder for `zone`
    sensor_type_encoder.joblib  - fitted LabelEncoder for `sensor_type`
    kmeans.joblib                - trained K-Means clustering model
    cluster_scaler.joblib       - fitted StandardScaler used before K-Means
    training_metrics.json       - metrics from this training run, timestamped

Run:
    python train_models.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, silhouette_score,
)

from schema import ZONES, SENSOR_TYPES, CLASSIFIER_FEATURES, CLUSTER_FEATURES
from data_generator import generate

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def train_classifier(df):
    print("\n--- Training Random Forest classifier ---")

    # Fit encoders on the FULL known category lists from schema.py, not just
    # whatever happens to appear in this training batch — otherwise a zone
    # that's rare in training but shows up live would break at inference.
    zone_encoder = LabelEncoder().fit(ZONES)
    sensor_encoder = LabelEncoder().fit(SENSOR_TYPES)

    df = df.copy()
    df["zone_enc"] = zone_encoder.transform(df["zone"])
    df["sensor_type_enc"] = sensor_encoder.transform(df["sensor_type"])

    X = df[CLASSIFIER_FEATURES]
    y = df["is_anomaly"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    for k, v in metrics.items():
        print(f"  {k:10s}: {v}")

    joblib.dump(clf, MODELS_DIR / "rf_classifier.joblib")
    joblib.dump(zone_encoder, MODELS_DIR / "zone_encoder.joblib")
    joblib.dump(sensor_encoder, MODELS_DIR / "sensor_type_encoder.joblib")

    return metrics


def train_clustering(df):
    print("\n--- Training K-Means clustering model ---")

    normal_df = df[df["is_anomaly"] == 0].copy()
    normal_df["reading_value_z"] = normal_df.groupby("sensor_type")["reading_value"] \
        .transform(lambda s: (s - s.mean()) / s.std())

    X = normal_df[CLUSTER_FEATURES].values
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    best_k, best_score, best_model = None, -1, None
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        score = silhouette_score(X_scaled, km.labels_, sample_size=3000, random_state=42)
        print(f"  k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score, best_model = k, score, km

    print(f"  Chosen k={best_k} (silhouette={best_score:.4f})")

    joblib.dump(best_model, MODELS_DIR / "kmeans.joblib")
    joblib.dump(scaler, MODELS_DIR / "cluster_scaler.joblib")

    return {"k": best_k, "silhouette_score": round(best_score, 4)}


def main():
    print("Generating historical training data...")
    df = generate(n_readings=20000)
    print(f"  {df.shape[0]} rows, {df['is_anomaly'].mean()*100:.1f}% anomalous")

    clf_metrics = train_classifier(df)
    cluster_metrics = train_clustering(df)

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_training_rows": len(df),
        "classifier": clf_metrics,
        "clustering": cluster_metrics,
    }
    with open(MODELS_DIR / "training_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nSaved models to ./models/:")
    for p in sorted(MODELS_DIR.glob("*")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
