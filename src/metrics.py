import re
import json
import platform
import psutil
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import src.config as config
from src.config import ModelConfig

# Canonical anomaly labels for extract_label()
UCF_ANOMALY_CATEGORIES = {
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism"
}
ANOMALY_LABELS_UPPER = {cat.upper() for cat in UCF_ANOMALY_CATEGORIES}


def extract_label(prediction_raw: str) -> str:
    """Map raw model output → 'Anomaly' | 'Normal' | 'UNKNOWN'."""
    match = re.search(r"[A-Za-z]+", prediction_raw)
    if not match:
        return "UNKNOWN"

    word = match.group(0).upper()
    if word == "NORMAL":
        return "Normal"
    if word in ANOMALY_LABELS_UPPER:
        return "Anomaly"

    text_upper = prediction_raw.upper()
    if any(label in text_upper for label in ANOMALY_LABELS_UPPER):
        return "Anomaly"
    if "NORMAL" in text_upper:
        return "Normal"

    return "UNKNOWN"


def save_experiment_config(cfg: ModelConfig, num_samples: int, output_prefix: str) -> None:
    """Save a JSON sidecar with all fixed experiment parameters."""
    conf = {
        "model_tag":          cfg.model_tag,
        "model_path":         cfg.model_path,
        "num_samples":        num_samples,
        "num_frames":         cfg.num_frames,
        "temperature":        cfg.temperature,
        "max_tokens":         cfg.max_tokens,
        "timestamp_start":    datetime.now().isoformat(),
        "platform":           platform.platform(),
        "python_version":     platform.python_version(),
        "cpu_count_logical":  psutil.cpu_count(logical=True),
        "ram_total_gb":       round(psutil.virtual_memory().total / 1024**3, 2),
    }
    path = output_prefix + "_config.json"
    with open(path, "w") as f:
        json.dump(conf, f, indent=2)
    print(f"📝 Config saved to: {path}")


def calculate_full_metrics(cfg: ModelConfig, results: list[dict], output_prefix: str) -> dict:
    """
    Compute and save classification + performance metrics.
    Anomaly = positive class (1), Normal = negative class (0).
    UNKNOWN predictions are treated as wrong (mapped to 0).
    """
    label_to_int = {"Anomaly": 1, "Normal": 0, "UNKNOWN": -1}

    y_true     = [label_to_int[r["ground_truth"]]              for r in results]
    y_pred_raw = [label_to_int.get(r["prediction_label"], -1)  for r in results]
    categories = [r["category"]                                 for r in results]

    unknown_count = sum(1 for p in y_pred_raw if p == -1)
    unknown_pct   = round(unknown_count / len(y_pred_raw) * 100, 2)

    # UNKNOWN → 0 (worst-case: treated as Normal, maximises FNR penalty)
    y_pred = [p if p != -1 else 0 for p in y_pred_raw]

    # --- overall binary metrics ---
    cm             = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    accuracy  = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0

    # --- per-category metrics ---
    cat_stats = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0})
    for gt, pred, cat in zip(y_true, y_pred, categories):
        cat_stats[cat]["total"] += 1
        if   gt == 1 and pred == 1: cat_stats[cat]["tp"] += 1
        elif gt == 0 and pred == 0: cat_stats[cat]["tn"] += 1
        elif gt == 0 and pred == 1: cat_stats[cat]["fp"] += 1
        elif gt == 1 and pred == 0: cat_stats[cat]["fn"] += 1

    per_category = {}
    for cat, s in cat_stats.items():
        p   = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else None
        r   = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else None
        f   = (2 * p * r / (p + r))         if (p and r)            else None
        acc = (s["tp"] + s["tn"]) / s["total"]                if s["total"] > 0  else None
        cfpr = s["fp"] / (s["fp"] + s["tn"]) if (s["fp"] + s["tn"]) > 0 else None
        cfnr = s["fn"] / (s["fn"] + s["tp"]) if (s["fn"] + s["tp"]) > 0 else None
        per_category[cat] = {
            "total":     s["total"],
            "tp": s["tp"], "tn": s["tn"], "fp": s["fp"], "fn": s["fn"],
            "accuracy":  round(acc,  4) if acc  is not None else None,
            "precision": round(p,    4) if p    is not None else None,
            "recall":    round(r,    4) if r    is not None else None,
            "f1":        round(f,    4) if f    is not None else None,
            "fpr":       round(cfpr, 4) if cfpr is not None else None,
            "fnr":       round(cfnr, 4) if cfnr is not None else None,
        }

    # --- performance summary ---
    infer_times = [r["inference_time_s"] for r in results if r.get("inference_time_s")]
    tps_values  = [r["tokens_per_sec"]   for r in results if r.get("tokens_per_sec")]

    metrics = {
        "model_tag":    cfg.model_tag,
        "num_samples":  len(results),
        "unknown_count": unknown_count,
        "unknown_pct":   unknown_pct,
        "overall": {
            "accuracy":    round(accuracy,  4),
            "precision":   round(precision, 4),
            "recall_tpr":  round(recall,    4),
            "f1":          round(f1,        4),
            "fpr":         round(fpr,       4),
            "fnr":         round(fnr,       4),
        },
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
        "per_category": per_category,
        "performance": {
            "avg_inference_time_s": round(sum(infer_times) / len(infer_times), 3) if infer_times else None,
            "avg_tokens_per_sec":   round(sum(tps_values)  / len(tps_values),  2) if tps_values  else None,
        },
    }

    # --- save ---
    metrics_file = output_prefix + "_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # --- print summary ---
    print("\n" + "=" * 55)
    print("📊 FULL METRICS REPORT")
    print("=" * 55)
    print(f"  Samples: {len(results)}  |  Unknown: {unknown_count} ({unknown_pct}%)")
    print(f"\n  {'Metric':<14} {'Value':>8}")
    print(f"  {'-' * 24}")
    for k, v in metrics["overall"].items():
        print(f"  {k:<14} {v:>8.4f}")
    print(f"\n  Confusion Matrix:  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  FPR (false alarm)  = {fpr:.4f}  ← Normal → Anomaly")
    print(f"  FNR (missed event) = {fnr:.4f}  ← Anomaly → Normal")
    print(f"\n  Avg inference: {metrics['performance']['avg_inference_time_s']}s  |  "
          f"Avg tok/s: {metrics['performance']['avg_tokens_per_sec']}")
    print("=" * 55)
    print(f"\n✅ Metrics saved to: {metrics_file}")

    return metrics