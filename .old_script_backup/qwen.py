import cv2
import csv
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from mlx_vlm import load, generate
from PIL import Image
import random
from collections import defaultdict
import re
import time
import json
import platform
import psutil
import os
import io
from contextlib import redirect_stdout
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import json

# --- CONFIG ---
MODEL_PATH = "mlx-community/Qwen3-VL-4B-Instruct-3bit"
DATA_PATH = "/Users/jakubmatkowski/Dokumenty/PW_repos/Praca_Badawcza/VLM-Test/UCF-Crime-Subset"
NUM_TEST_SAMPLES = 20
NUM_OF_FRAMES = 16
TEMPERATURE = 0.0
MAX_TOKENS = 50
# Output file name encodes key config — makes multi-model comparison easy
MODEL_TAG = "Qwen3VL4B_3bit"
OUTPUT_FILE = f"results_{MODEL_TAG}_{NUM_OF_FRAMES}frames_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
CONFIG_FILE = OUTPUT_FILE.replace(".csv", "_config.json")

model, processor = load(MODEL_PATH)

UCF_ANOMALY_CATEGORIES = {
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism"
}
ANOMALY_LABELS_UPPER = {cat.upper() for cat in UCF_ANOMALY_CATEGORIES}

# psutil process handle — reused across calls
_process = psutil.Process(os.getpid())

def get_video_files(base_path):
    base = Path(base_path)
    dataset = []

    for video_path in (base / "Anomaly-Videos-Part-1").rglob("*.mp4"):
        dataset.append({
            "path": video_path,
            "label": "Anomaly",
            "category": video_path.parent.name
        })

    for video_path in (base / "Normal_Videos_for_Event_Recognition").rglob("*.mp4"):
        dataset.append({
            "path": video_path,
            "label": "Normal",
            "category": "Normal"
        })
    return dataset

def extract_frames(video_path, num_frames):
    """Returns (frames, total_frames, fps, duration_s) or ([], 0, 0, 0) on failure."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    duration_s = round(total_frames / fps, 2)

    if total_frames <= 0:
        cap.release()
        return [], 0, 0, 0

    frame_indices = [int(total_frames * i / num_frames) for i in range(num_frames)]

    frames = []
    for frame_id in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames, total_frames, fps, duration_s

def extract_label(prediction_raw: str) -> str:
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

def save_experiment_config(num_samples):
    """Save a JSON sidecar with all fixed experiment parameters."""
    config = {
        "model_path": MODEL_PATH,
        "model_tag": MODEL_TAG,
        "num_samples": num_samples,
        "num_frames": NUM_OF_FRAMES,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "timestamp_start": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "output_file": OUTPUT_FILE,
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"📝 Config saved to: {CONFIG_FILE}")

def calculate_accuracy(results):
    print("\n" + "=" * 50)
    print("📊 CLASSIFICATION REPORT")
    print("=" * 50)

    per_category = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = overall_total = 0

    for result in results:
        cat = result["category"]
        predicted = result["prediction_label"]
        gt = result["ground_truth"]
        per_category[cat]["total"] += 1
        overall_total += 1
        if predicted == gt:
            per_category[cat]["correct"] += 1
            overall_correct += 1

    for cat in sorted(per_category.keys()):
        m = per_category[cat]
        acc = m["correct"] / m["total"] * 100 if m["total"] else 0
        print(f"  {cat:20} {m['correct']}/{m['total']} ({acc:5.1f}%)")

    overall_acc = overall_correct / overall_total * 100 if overall_total else 0
    print("=" * 50)
    print(f"  {'OVERALL':20} {overall_correct}/{overall_total} ({overall_acc:5.1f}%)")
    print("=" * 50)

def analyze_video_with_vlm(frames, custom_prompt):
    pil_images = [Image.fromarray(f) for f in frames]
    enhanced_prompt = custom_prompt + "\n\nRespond with ONLY: [LABEL] - reason (max 20 words)"
    content = [{"type": "image"} for _ in pil_images]
    content.append({"type": "text", "text": enhanced_prompt})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Przechwytujemy stdout żeby wyłuskać tokens/sec
    buf = io.StringIO()
    with redirect_stdout(buf):
        response = generate(
            model, processor,
            prompt=prompt, image=pil_images,
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            verbose=True
        )
    
    # mlx_vlm drukuje coś w stylu: "Prompt: 14 tokens, Generation: 12 tokens, 23.4 tokens/sec"
    tps = None
    for line in buf.getvalue().splitlines():
        match = re.search(r"([\d.]+)\s*tokens/sec", line)
        if match:
            tps = float(match.group(1))
            break

    text = response if isinstance(response, str) else response.text
    return text, tps

def run_experiment(video_list, num_samples):
    results = []
    test_subset = video_list[:num_samples]
    print(f"🚀 Starting test. {len(test_subset)} videos to process.\n")

    prompt = """You are an advanced security analyst. Analyze these CCTV frames.
    ARSON: Fire, smoke, ignition.
    ASSAULT: Physical fight, hitting, grappling.
    ABUSE: Physical domination, restraining, intimidation.
    ARREST: Police restraining/handcuffing someone.
    NORMAL: Peaceful activity, walking, no violence."""

    for item in tqdm(test_subset, desc="Processing videos"):

        # ── 1. Frame extraction (timed) ──────────────────────────────────
        t0 = time.time()
        frames, total_frames, fps, duration_s = extract_frames(item["path"], NUM_OF_FRAMES)
        frame_time = round(time.time() - t0, 3)

        if not frames:
            continue

        # ── 2. Inference (timed + memory delta) ──────────────────────────
        mem_before = _process.memory_info().rss / 1024**2          # MB
        t1 = time.time()
        pred, tokens_per_sec = analyze_video_with_vlm(frames, prompt)
        infer_time = round(time.time() - t1, 3)
        mem_after = _process.memory_info().rss / 1024**2            # MB

        pred = pred.strip()
        pred_label = extract_label(pred)

        try:
            res_entry = {
                # ── identification ──
                "timestamp":               datetime.now().strftime("%H:%M:%S"),
                "filename":                item["path"].name,
                "category":                item["category"],
                "model_tag":               MODEL_TAG,          # ← which model ran this

                # ── ground truth & prediction ──
                "ground_truth":            item["label"],
                "prediction_raw":          pred,
                "prediction_label":        pred_label,
                "prediction_correct":      pred_label == item["label"],
                "is_unknown":              pred_label == "UNKNOWN",
                "response_word_count":     len(pred.split()),  # proxy for model "confidence"

                # ── video metadata ──
                "video_duration_s":        duration_s,
                "video_total_frames":      total_frames,
                "video_fps":               round(fps, 2),
                "frames_sampled":          len(frames),

                # ── performance ──
                "inference_time_s":        infer_time,
                "tokens_per_sec":          tokens_per_sec,
                "frame_extraction_time_s": frame_time,
                "mem_delta_mb":            round(mem_after - mem_before, 1),  # RAM spike per inference
            }
            results.append(res_entry)

            with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=res_entry.keys())
                writer.writeheader()
                writer.writerows(results)

        except Exception as e:
            print(f"\n❌ Error {item['path'].name}: {e}")

    return results

def calculate_full_metrics(results, output_prefix):
    """
    Liczy i zapisuje kompletny raport metryk do JSON.
    Wyniki z prediction_label == 'UNKNOWN' są traktowane jako błędne predykcje.
    """
    # Zamień etykiety na binarne 0/1
    # Anomaly = 1 (positive class w surveillance), Normal = 0
    label_to_int = {"Anomaly": 1, "Normal": 0, "UNKNOWN": -1}

    y_true = []
    y_pred = []
    categories = []

    for r in results:
        y_true.append(label_to_int[r["ground_truth"]])
        y_pred.append(label_to_int.get(r["prediction_label"], -1))
        categories.append(r["category"])

    # --- % UNKNOWN ---
    unknown_count = sum(1 for p in y_pred if p == -1)
    unknown_pct = round(unknown_count / len(y_pred) * 100, 2)

    # Do metryk binarnych: UNKNOWN traktujemy jako błąd (nie pasuje do żadnej klasy)
    # Zamieniamy -1 na 0 (Normal) — najgorszy przypadek dla FNR
    y_pred_binary = [p if p != -1 else 0 for p in y_pred]
    y_true_binary = y_true  # już są 0/1

    # --- CONFUSION MATRIX ---
    # [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    # --- OVERALL BINARY METRICS ---
    precision  = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall     = recall_score(y_true_binary, y_pred_binary, zero_division=0)   # = TPR
    f1         = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    accuracy   = sum(t == p for t, p in zip(y_true_binary, y_pred_binary)) / len(y_true_binary)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate (fałszywy alarm)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate (przeoczenie)

    # --- PER-CATEGORY METRICS ---
    from collections import defaultdict
    cat_stats = defaultdict(lambda: {"tp":0,"tn":0,"fp":0,"fn":0,"total":0})

    for gt, pred, cat in zip(y_true_binary, y_pred_binary, categories):
        cat_stats[cat]["total"] += 1
        if gt == 1 and pred == 1: cat_stats[cat]["tp"] += 1
        elif gt == 0 and pred == 0: cat_stats[cat]["tn"] += 1
        elif gt == 0 and pred == 1: cat_stats[cat]["fp"] += 1
        elif gt == 1 and pred == 0: cat_stats[cat]["fn"] += 1

    per_category = {}
    for cat, s in cat_stats.items():
        cat_precision = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else None
        cat_recall    = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else None
        cat_f1 = (2 * cat_precision * cat_recall / (cat_precision + cat_recall)
                  if cat_precision and cat_recall else None)
        cat_acc = (s["tp"] + s["tn"]) / s["total"] if s["total"] > 0 else None
        cat_fpr = s["fp"] / (s["fp"] + s["tn"]) if (s["fp"] + s["tn"]) > 0 else None
        cat_fnr = s["fn"] / (s["fn"] + s["tp"]) if (s["fn"] + s["tp"]) > 0 else None
        per_category[cat] = {
            "total": s["total"], "tp": s["tp"], "tn": s["tn"],
            "fp": s["fp"], "fn": s["fn"],
            "accuracy":  round(cat_acc, 4)       if cat_acc  is not None else None,
            "precision": round(cat_precision, 4) if cat_precision is not None else None,
            "recall":    round(cat_recall, 4)    if cat_recall is not None else None,
            "f1":        round(cat_f1, 4)        if cat_f1   is not None else None,
            "fpr":       round(cat_fpr, 4)       if cat_fpr  is not None else None,
            "fnr":       round(cat_fnr, 4)       if cat_fnr  is not None else None,
        }

    # --- PERFORMANCE SUMMARY ---
    infer_times = [r["inference_time_s"] for r in results if r.get("inference_time_s")]
    tps_values  = [r["tokens_per_sec"]   for r in results if r.get("tokens_per_sec")]

    metrics = {
        "model_tag": MODEL_TAG,
        "num_samples": len(results),
        "unknown_count": unknown_count,
        "unknown_pct": unknown_pct,
        "overall": {
            "accuracy":  round(accuracy, 4),
            "precision": round(precision, 4),
            "recall_tpr": round(recall, 4),
            "f1":        round(f1, 4),
            "fpr":       round(fpr, 4),
            "fnr":       round(fnr, 4),
        },
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
        "per_category": per_category,
        "performance": {
            "avg_inference_time_s": round(sum(infer_times)/len(infer_times), 3) if infer_times else None,
            "avg_tokens_per_sec":   round(sum(tps_values)/len(tps_values), 2)  if tps_values else None,
        }
    }

    # Zapis do JSON
    metrics_file = f"{output_prefix}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Czytelny print
    print("\n" + "="*55)
    print("📊 FULL METRICS REPORT")
    print("="*55)
    print(f"  Samples: {len(results)}  |  Unknown: {unknown_count} ({unknown_pct}%)")
    print(f"\n  {'Metric':<12} {'Value':>8}")
    print(f"  {'-'*22}")
    for k, v in metrics["overall"].items():
        print(f"  {k:<12} {v:>8.4f}")
    print(f"\n  Confusion Matrix:  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  FPR (false alarm)  = {fpr:.4f}  ← jak często Normal → Anomaly")
    print(f"  FNR (missed event) = {fnr:.4f}  ← jak często Anomaly → Normal")
    print(f"\n  Avg inference: {metrics['performance']['avg_inference_time_s']}s  |  "
          f"Avg tok/s: {metrics['performance']['avg_tokens_per_sec']}")
    print("="*55)
    print(f"\n✅ Metrics saved to: {metrics_file}")

    return metrics

# --- EXECUTION ---
if __name__ == "__main__":
    videos = get_video_files(DATA_PATH)
    print(f"📁 Found {len(videos)} files in dataset.\n")

    save_experiment_config(NUM_TEST_SAMPLES)  # ← zapisz config przed startem

    random.shuffle(videos)
    results_data = run_experiment(videos, num_samples=NUM_TEST_SAMPLES)

    if results_data:
        calculate_accuracy(results_data)
        calculate_full_metrics(results_data, output_prefix=OUTPUT_FILE.replace(".csv", ""))