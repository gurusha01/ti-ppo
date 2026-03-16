"""Generate all GSM8K benchmark plots.

Reads results from eval_gsm8k_results_*.json and sweep_epsilon_results_*.json
and generates comprehensive plots.
"""

import json, os, glob, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def smooth(y, w=15):
    y = np.array(y, dtype=float)
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode='valid')


def load_eval_results():
    """Load eval results from eval_gsm8k_results_*.json"""
    results = {}
    for fn in sorted(glob.glob(os.path.join(PROJECT_ROOT, "logs", "eval", "eval_gsm8k_results_*.json"))):
        with open(fn) as f:
            d = json.load(f)
        label = d.get("label", d.get("method", os.path.basename(fn)))
        results[label] = d
    return results


def load_benchmark_histories():
    """Load training histories from benchmark_gsm8k_results_*_*.json"""
    histories = {}
    for fn in sorted(glob.glob(os.path.join(PROJECT_ROOT, "logs", "benchmark", "benchmark_gsm8k_results_*_*.json"))):
        with open(fn) as f:
            d = json.load(f)
        label = d["results"][0]["label"]
        histories[label] = d["history"]
    return histories


def load_epsilon_sweep():
    """Load epsilon sweep results."""
    results = []
    for fn in sorted(glob.glob(os.path.join(PROJECT_ROOT, "logs", "sweep_epsilon", "sweep_epsilon_results_*.json"))):
        with open(fn) as f:
            d = json.load(f)
        results.append(d)
    return results


def load_batchsize_sweep():
    """Load batch size sweep results."""
    results = []
    for fn in sorted(glob.glob(os.path.join(PROJECT_ROOT, "sweep_batchsize_results_*.json"))):
        with open(fn) as f:
            d = json.load(f)
        results.append(d)
    return results


COLORS = [
    "#e74c3c", "#3498db", "#27ae60", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#888888",
    "#2ecc71", "#c0392b",
]


def plot_training_curves(histories):
    """4-panel training curves: reward, KL, accuracy, importance."""
    if not histories:
        print("No training histories found, skipping training curves")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("GSM8K Training Curves — TI-PPO Methods", fontsize=15, fontweight='bold')

    for idx, (label, h) in enumerate(histories.items()):
        c = COLORS[idx % len(COLORS)]

        # Accuracy
        axes[0, 0].plot(h["accuracy"], label=label, color=c, linewidth=1.3, alpha=0.85)

        # Reward
        r = smooth(h["rewards"])
        axes[0, 1].plot(range(len(r)), r, label=label, color=c, linewidth=1.3, alpha=0.85)

        # KL
        kl = smooth(h["kl"])
        axes[1, 0].plot(range(len(kl)), kl, label=label, color=c, linewidth=1.3, alpha=0.85)

        # Importance
        if label != "PPO baseline":
            imp = smooth(h["importance"])
            axes[1, 1].plot(range(len(imp)), imp, label=label, color=c, linewidth=1.3, alpha=0.85)

    for ax, title, ylabel in [
        (axes[0, 0], "Running Accuracy", "Accuracy"),
        (axes[0, 1], "Reward (smoothed)", "Reward"),
        (axes[1, 0], "KL Divergence (smoothed)", "KL"),
        (axes[1, 1], "Importance Weights (smoothed)", "Mean Importance"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    axes[0, 1].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5, linestyle='--')

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "gsm8k_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def plot_test_accuracy_bar(eval_results):
    """Bar chart of test set accuracy."""
    if not eval_results:
        print("No eval results found, skipping test accuracy bar chart")
        return

    labels = list(eval_results.keys())
    accs = [eval_results[l].get("test_accuracy", eval_results[l].get("accuracy", 0)) * 100
            for l in labels]

    # Sort by accuracy
    paired = sorted(zip(labels, accs), key=lambda x: -x[1])
    labels, accs = zip(*paired)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(labels)), accs, color=[COLORS[i % len(COLORS)] for i in range(len(labels))])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Test Accuracy (%)")
    ax.set_title("GSM8K Test Set Accuracy (1319 examples, greedy decoding)")
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{acc:.1f}%', va='center', fontsize=9)

    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "gsm8k_test_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def plot_epsilon_sweep(sweep_results):
    """Plot epsilon sweep: accuracy vs fixed epsilon, with MOAI adaptive marked."""
    if not sweep_results:
        print("No epsilon sweep results found, skipping")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Epsilon Sweep: Fixed vs Adaptive (MOAI)", fontsize=14, fontweight='bold')

    fixed = [r for r in sweep_results if "Fixed" in r.get("label", "")]
    adaptive = [r for r in sweep_results if "adaptive" in r.get("label", "").lower() or "MOAI" in r.get("label", "")]

    # Extract epsilon values and metrics
    fixed_eps = []
    fixed_acc = []
    fixed_kl = []
    for r in fixed:
        label = r.get("label", "")
        # Parse epsilon from label like "Fixed eps=0.3"
        import re
        m = re.search(r"eps=([\d.]+)", label)
        if m:
            fixed_eps.append(float(m.group(1)))
            fixed_acc.append(r.get("final_accuracy", 0))
            kl_vals = r.get("history", {}).get("kl", [])
            fixed_kl.append(np.mean(kl_vals[-50:]) if kl_vals else 0)

    # Sort by epsilon
    if fixed_eps:
        order = np.argsort(fixed_eps)
        fixed_eps = [fixed_eps[i] for i in order]
        fixed_acc = [fixed_acc[i] for i in order]
        fixed_kl = [fixed_kl[i] for i in order]

    # Accuracy vs epsilon
    ax = axes[0]
    if fixed_eps:
        ax.plot(fixed_eps, fixed_acc, 'bo-', label='Fixed epsilon', linewidth=2, markersize=8)
    for r in adaptive:
        acc = r.get("final_accuracy", 0)
        ax.axhline(y=acc, color='red', linestyle='--', linewidth=2, label=f'{r.get("label", "MOAI")} ({acc:.3f})')
    ax.set_xlabel("Epsilon (importance intensity)")
    ax.set_ylabel("Final Training Accuracy")
    ax.set_title("Accuracy vs Epsilon")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # KL vs epsilon
    ax = axes[1]
    if fixed_eps:
        ax.plot(fixed_eps, [abs(k) for k in fixed_kl], 'bo-', label='Fixed epsilon', linewidth=2, markersize=8)
    for r in adaptive:
        kl_vals = r.get("history", {}).get("kl", [])
        kl = abs(np.mean(kl_vals[-50:])) if kl_vals else 0
        ax.axhline(y=kl, color='red', linestyle='--', linewidth=2, label=f'{r.get("label", "MOAI")} ({kl:.4f})')
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("|KL| (last 50 episodes)")
    ax.set_title("KL Divergence vs Epsilon")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Training curves for all epsilon values
    ax = axes[2]
    for r in sweep_results:
        acc = r.get("history", {}).get("accuracy", [])
        if acc:
            style = '--' if "MOAI" in r.get("label", "") else '-'
            ax.plot(acc, label=r.get("label", "?")[:20], linewidth=1.2, linestyle=style)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Running Accuracy")
    ax.set_title("Learning Curves by Epsilon")
    ax.legend(fontsize=6, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "gsm8k_epsilon_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def plot_pareto(histories):
    """Pareto frontier: accuracy vs |KL|."""
    if not histories:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Pareto Frontier: Accuracy vs |KL|", fontsize=14, fontweight='bold')

    for idx, (label, h) in enumerate(histories.items()):
        n = len(h["accuracy"])
        q4_start = 3 * n // 4
        acc_q4 = np.mean(h["accuracy"][q4_start:])
        kl_q4 = np.mean(np.abs(h["kl"][q4_start:]))
        c = COLORS[idx % len(COLORS)]
        ax.scatter(kl_q4, acc_q4, s=150, color=c, zorder=5, edgecolors='black', linewidth=0.5)
        ax.annotate(label, (kl_q4, acc_q4), textcoords="offset points",
                    xytext=(8, 5), fontsize=8)

    ax.set_xlabel("|KL| (Q4 average)")
    ax.set_ylabel("Accuracy (Q4 average)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "gsm8k_pareto.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    print("Loading data...")
    histories = load_benchmark_histories()
    eval_results = load_eval_results()
    epsilon_sweep = load_epsilon_sweep()

    print(f"Found {len(histories)} training histories, {len(eval_results)} eval results, {len(epsilon_sweep)} epsilon sweep results")

    plot_training_curves(histories)
    plot_test_accuracy_bar(eval_results)
    plot_epsilon_sweep(epsilon_sweep)
    plot_pareto(histories)
    print("\nDone! All plots saved to", PLOT_DIR)
