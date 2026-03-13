import csv, time, random, psutil, os
from datetime import datetime
from tqdm import tqdm

import src.config as config
from src.config import QWEN_CONFIG, SMOLVLM_CONFIG, IDEFICS_CONFIG, GEMMA4B_CONFIG, GEMMA3N2B_CONFIG
from src.dataset import get_video_files, extract_frames
from src.metrics import extract_label, save_experiment_config, calculate_full_metrics
from models.qwen_model import QwenModel
from models.smolvlm_model import SmolVLMModel
from models.idefics_model import IdeficsModel
from models.gemma4b_model import Gemma4bModel
from models.gemma3nE2B import Gemma3nE2BModel

_process = psutil.Process(os.getpid())

#==== Model Selection ==========================================================
ACTIVE_CONFIG = GEMMA3N2B_CONFIG
ACTIVE_MODEL  = Gemma3nE2BModel(ACTIVE_CONFIG)
#===============================================================================


def run_experiment(model, video_list: list, num_samples: int) -> tuple[list, str]:
    results       = []
    test_subset   = video_list[:num_samples]
    output_prefix = config.make_output_prefix(model.cfg)
    output_file   = output_prefix + ".csv"

    print(f"🚀 Starting test. {len(test_subset)} videos to process.\n")

    for item in tqdm(test_subset, desc="Processing videos..."):
        try:
            t0 = time.time()
            frames, total_frames, fps, duration_s = extract_frames(
                item["path"],
                model.cfg.num_frames
            )
            frame_time = round(time.time() - t0, 3)

            if not frames:
                continue

            mem_before = _process.memory_info().rss / 1024**2
            t1 = time.time()
            pred, tokens_per_sec = model.analyze(frames, model.cfg.prompt)
            infer_time = round(time.time() - t1, 3)
            mem_after  = _process.memory_info().rss / 1024**2

            pred       = pred.strip()
            pred_label = extract_label(pred)

            res_entry = {
                "timestamp":               datetime.now().strftime("%H:%M:%S"),
                "filename":                item["path"].name,
                "category":                item["category"],
                "model_tag":               model.cfg.model_tag,
                "ground_truth":            item["label"],
                "prediction_raw":          pred,
                "prediction_label":        pred_label,
                "prediction_correct":      pred_label == item["label"],
                "is_unknown":              pred_label == "UNKNOWN",
                "response_word_count":     len(pred.split()),
                "video_duration_s":        duration_s,
                "video_total_frames":      total_frames,
                "video_fps":               round(fps, 2),
                "frames_sampled":          len(frames),
                "inference_time_s":        infer_time,
                "tokens_per_sec":          tokens_per_sec,
                "frame_extraction_time_s": frame_time,
                "mem_delta_mb":            round(mem_after - mem_before, 1),
            }
            results.append(res_entry)

            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=res_entry.keys())
                writer.writeheader()
                writer.writerows(results)

        except Exception as e:
            print(f"\nError {item['path'].name}: {e}")

    return results, output_prefix


if __name__ == "__main__":
    videos = get_video_files(config.DATA_PATH)
    print(f"Found {len(videos)} files.\n")

    ACTIVE_MODEL.load()

    output_prefix = config.make_output_prefix(ACTIVE_MODEL.cfg)
    save_experiment_config(ACTIVE_MODEL.cfg, config.NUM_TEST_SAMPLES, output_prefix)

    random.shuffle(videos)
    results, output_prefix = run_experiment(ACTIVE_MODEL, videos, config.NUM_TEST_SAMPLES)

    if results:
        calculate_full_metrics(ACTIVE_MODEL.cfg, results, output_prefix)