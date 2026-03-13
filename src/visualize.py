import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path("results")
PLOTS_DIR   = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Kolory per model (dodaj kolejne jeśli dojdą nowe) ─────────────────────────
MODEL_COLORS = {
    "Qwen3VL4B_3bit":          "#4C72B0",
    "gemma-3-4b-it-4bit":      "#DD8452",
    "gemma-3-1b-it-4bit":      "#55A868",
    "SmolVLM2_2.2B_mlx":       "#C44E52",
    "Idefics3-8B-Llama3-4bit": "#8172B2",
}
FALLBACK_COLORS = plt.cm.tab10.colors


# ══════════════════════════════════════════════════════════════════════════════
# 1. WCZYTYWANIE DANYCH
# ══════════════════════════════════════════════════════════════════════════════

def load_all_metrics(results_dir: Path) -> list[dict]:
    """
    Wczytuje każdy plik *_metrics.json z results/.
    Jeśli dla jednego modelu jest kilka plików (różne daty),
    bierze najnowszy na podstawie nazwy pliku.
    """
    files = sorted(results_dir.glob("*_metrics.json"))
    if not files:
        raise FileNotFoundError(f"Brak plików *_metrics.json w {results_dir}")

    # Grupuj po model_tag — bierz najnowszy plik
    latest: dict[str, Path] = {}
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        tag = data.get("model_tag", f.stem)
        # Późniejszy plik w sorted() zawsze zastępuje wcześniejszy
        latest[tag] = f

    metrics = []
    for tag, path in latest.items():
        with open(path) as fp:
            data = json.load(fp)
        data["_file"] = path.name
        metrics.append(data)
        print(f" {tag:35} ← {path.name}")

    return metrics


def get_color(model_tag: str, idx: int) -> str:
    return MODEL_COLORS.get(model_tag, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])

def plot_overall_metrics(metrics: list[dict]) -> None:
    """
    Grouped bar chart: Accuracy / Precision / Recall / F1 per model.
    """
    keys   = ["accuracy", "precision", "recall_tpr", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    n_models = len(metrics)
    n_metrics = len(keys)

    x      = np.arange(n_metrics)
    width  = 0.8 / n_models          # szerokość jednego słupka

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, m in enumerate(metrics):
        values = [m["overall"].get(k, 0) for k in keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(x + offset, values, width, label=m["model_tag"],
                        color=get_color(m["model_tag"], i), alpha=0.88)
        # wartości nad słupkami
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Overall Classification Metrics — Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = PLOTS_DIR / "01_overall_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f" {out}")


def plot_fpr_fnr(metrics: list[dict]) -> None:
    """
    Dwa osobne słupki: FPR (fałszywy alarm) i FNR (przeoczenie).
    Kontekst surveillance: oba powinny być jak najniższe.
    """
    tags = [m["model_tag"] for m in metrics]
    fpr  = [m["overall"]["fpr"] for m in metrics]
    fnr  = [m["overall"]["fnr"] for m in metrics]
    colors = [get_color(m["model_tag"], i) for i, m in enumerate(metrics)]

    x     = np.arange(len(tags))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - width / 2, fpr, width, label="FPR (false alarm)",
                color=colors, alpha=0.9, hatch="//")
    b2 = ax.bar(x + width / 2, fnr, width, label="FNR (missed event)",
                color=colors, alpha=0.6)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("FPR vs FNR — Surveillance Error Trade-off", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = PLOTS_DIR / "02_fpr_fnr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f" {out}")


def plot_confusion_matrices(metrics: list[dict]) -> None:
    """
    Siatka confusion matrix dla każdego modelu (binary: Anomaly/Normal).
    """
    n     = len(metrics)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()   # zawsze 1D

    for i, m in enumerate(metrics):
        cm   = m["confusion_matrix"]
        data = np.array([[cm["tn"], cm["fp"]],
                         [cm["fn"], cm["tp"]]])
        ax   = axes[i]
        im   = ax.imshow(data, cmap="Blues", vmin=0)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Normal", "Pred: Anomaly"], fontsize=9)
        ax.set_yticklabels(["True: Normal", "True: Anomaly"], fontsize=9)
        ax.set_title(m["model_tag"], fontsize=9, fontweight="bold")

        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(data[r, c]), ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if data[r, c] > data.max() / 2 else "black")

    # ukryj puste subploty
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = PLOTS_DIR / "03_confusion_matrices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" {out}")


def plot_per_category_heatmap(metrics: list[dict]) -> None:
    """
    Heatmapa F1 per kategoria per model.
    Oś X: modele, oś Y: kategorie UCF-Crime.
    """
    # Zbierz wszystkie kategorie
    all_cats = set()
    for m in metrics:
        all_cats.update(m["per_category"].keys())
    cats = sorted(all_cats)
    tags = [m["model_tag"] for m in metrics]

    # Macierz F1 [kategoria x model]
    matrix = np.full((len(cats), len(tags)), np.nan)
    for j, m in enumerate(metrics):
        for i, cat in enumerate(cats):
            cat_data = m["per_category"].get(cat, {})
            # Dla Normal: F1 nie istnieje (brak TP) — użyj accuracy
            val = cat_data.get("f1") or cat_data.get("accuracy")
            if val is not None:
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(max(8, len(tags) * 2), max(5, len(cats) * 0.7)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(tags)))
    ax.set_yticks(range(len(cats)))
    ax.set_xticklabels(tags, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(cats, fontsize=10)

    # wartości w komórkach
    for i in range(len(cats)):
        for j in range(len(tags)):
            val = matrix[i, j]
            txt = f"{val:.2f}" if not np.isnan(val) else "–"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="black" if 0.3 < val < 0.85 else "white")

    plt.colorbar(im, ax=ax, label="F1 Score (Accuracy for Normal)")
    ax.set_title("Per-Category F1 Heatmap — Model × Category", fontsize=13, fontweight="bold")

    fig.tight_layout()
    out = PLOTS_DIR / "04_per_category_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" {out}")


def plot_performance(metrics: list[dict]) -> None:
    """
    Scatter: inference time vs F1 — idealny model jest w lewym górnym rogu.
    Rozmiar punktu = liczba próbek.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, m in enumerate(metrics):
        t   = m["performance"].get("avg_inference_time_s")
        f1  = m["overall"]["f1"]
        tag = m["model_tag"]
        if t is None:
            continue
        color = get_color(tag, i)
        ax.scatter(t, f1, s=200, color=color, zorder=3, edgecolors="white", linewidths=1.5)
        ax.annotate(tag, (t, f1), textcoords="offset points",
                    xytext=(8, 4), fontsize=8.5, color=color)

    ax.set_xlabel("Avg Inference Time (s) — niżej = szybciej", fontsize=11)
    ax.set_ylabel("F1 Score — wyżej = lepiej", fontsize=11)
    ax.set_title("Speed vs Accuracy Trade-off", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    # Zaznacz idealny narożnik
    ax.annotate("← idealny model", xy=(ax.get_xlim()[0], 1.0),
                fontsize=9, color="gray", style="italic")

    fig.tight_layout()
    out = PLOTS_DIR / "05_speed_vs_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  📊 {out}")


def plot_unknown_pct(metrics: list[dict]) -> None:
    """
    % predykcji UNKNOWN per model — wskaźnik 'posłuszeństwa' wobec instrukcji.
    """
    tags    = [m["model_tag"] for m in metrics]
    values  = [m["unknown_pct"] for m in metrics]
    colors  = [get_color(m["model_tag"], i) for i, m in enumerate(metrics)]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(tags, values, color=colors, alpha=0.88)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, max(values or [10]) * 1.3 + 5)
    ax.set_ylabel("% UNKNOWN predictions", fontsize=11)
    ax.set_title("Instruction Following — % of UNKNOWN Responses", fontsize=13, fontweight="bold")
    ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=9)
    ax.axhline(0, color="green", linewidth=1, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = PLOTS_DIR / "06_unknown_pct.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  📊 {out}")

if __name__ == "__main__":
    print(f"\n Metrics loading from {RESULTS_DIR}/\n")
    all_metrics = load_all_metrics(RESULTS_DIR)

    print(f"\n Generating plots...→ {PLOTS_DIR}/\n")
    plot_overall_metrics(all_metrics)
    plot_fpr_fnr(all_metrics)
    plot_confusion_matrices(all_metrics)
    plot_per_category_heatmap(all_metrics)
    plot_performance(all_metrics)
    plot_unknown_pct(all_metrics)

    print(f"\n Ready! {len(all_metrics)} models in {PLOTS_DIR}/")