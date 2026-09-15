"""Metrics owned by the engine, with prediction-ID alignment checks."""
import math


def evaluate(queries, predictions):
    truth = {q["id"]: q["target"] for q in queries}
    if len(truth) != len(queries):
        raise ValueError("Duplicate target IDs")
    ids = [p["id"] for p in predictions]
    if len(set(ids)) != len(ids) or set(ids) != set(truth):
        raise ValueError("Prediction IDs must exactly match evaluation IDs")
    confusion = [[0, 0], [0, 0]]
    for row in predictions:
        prob = row["probability"]
        if type(prob) not in (int, float) or not math.isfinite(prob) or not 0 <= prob <= 1:
            raise ValueError("Predictions must be finite probabilities")
        confusion[truth[row["id"]]][int(prob >= .5)] += 1
    if not ids:
        raise ValueError("Cannot evaluate empty predictions")
    f1 = []
    for c in (0, 1):
        tp = confusion[c][c]
        denominator = 2 * tp + confusion[c][1-c] + confusion[1-c][c]
        f1.append(2 * tp / denominator if denominator else 0.0)
    return {"macro_f1": sum(f1) / 2, "accuracy": sum(confusion[c][c] for c in (0, 1)) / len(ids),
            "confusion_matrix": confusion, "count": len(ids), "threshold": .5}
