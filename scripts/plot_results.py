"""Generate all research plots from benchmark JSON results."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f8f8',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'figure.dpi': 150,
})

def smooth(y, window=7):
    """Simple moving average."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')

def load(name):
    with open(ROOT / name) as f:
        return json.load(f)

# ─── Colors ───
COLORS = {
    'PPO baseline': '#666666',
    'AITI-Advantage (linear)': '#2196F3',
    'AITI-Entropy (linear)': '#4CAF50',
    'AITI-Entropy (quadratic)': '#8BC34A',
    'AITI-Entropy (residual 0.1)': '#CDDC39',
    'AITI-AdaptivePhase (linear)': '#00BCD4',
    'AITI-AdaptivePhase (quad)': '#009688',
    'Entropy (no decay)': '#FF9800',
    'Adaptive Phase (no decay)': '#FF5722',
    'MOAI-Adv mono (ema=0.80)': '#E91E63',
    'MOAI-Adv mono (ema=0.90)': '#9C27B0',
    'MOAI-Adv mono (ema=0.95)': '#673AB7',
    'MOAI-Adv free (ema=0.80)': '#F44336',
    'MOAI-Ent mono (ema=0.95)': '#795548',
}

def get_color(label):
    return COLORS.get(label, '#333333')

# ════════════════════════════════════════════════════════
# PLOT 1: Reward curves — v4 (AITI vs baselines)
# ════════════════════════════════════════════════════════
def plot_reward_curves_v4():
    d = load('benchmark_v4_results.json')
    fig, ax = plt.subplots(figsize=(12, 6))

    highlight = ['PPO baseline', 'AITI-Advantage (linear)', 'Entropy (no decay)',
                 'AITI-Entropy (quadratic)']

    for label, hist in d['histories'].items():
        r = np.array(hist['rewards'])
        rs = smooth(r, 9)
        x = np.arange(len(rs))
        alpha = 1.0 if label in highlight else 0.3
        lw = 2.5 if label in highlight else 1.0
        ax.plot(x, rs, label=label, color=get_color(label), alpha=alpha, linewidth=lw)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (smoothed)')
    ax.set_title('Phase 3: AITI vs Baselines — Reward Over Training')
    ax.legend(loc='lower right', fontsize=9)
    ax.axvline(x=100, color='red', linestyle='--', alpha=0.4, label='decay_steps=100')
    fig.tight_layout()
    fig.savefig(OUT / 'reward_curves_v4_aiti.png')
    plt.close(fig)
    print(f"Saved: {OUT / 'reward_curves_v4_aiti.png'}")

# ════════════════════════════════════════════════════════
# PLOT 2: KL curves — v4
# ════════════════════════════════════════════════════════
def plot_kl_curves_v4():
    d = load('benchmark_v4_results.json')
    fig, ax = plt.subplots(figsize=(12, 6))

    highlight = ['PPO baseline', 'AITI-Advantage (linear)', 'Entropy (no decay)']

    for label, hist in d['histories'].items():
        kl = np.array(hist['kl'])
        kls = smooth(kl, 9)
        x = np.arange(len(kls))
        alpha = 1.0 if label in highlight else 0.3
        lw = 2.5 if label in highlight else 1.0
        ax.plot(x, kls, label=label, color=get_color(label), alpha=alpha, linewidth=lw)

    ax.set_xlabel('Episode')
    ax.set_ylabel('KL Divergence (smoothed)')
    ax.set_title('Phase 3: KL Divergence Over Training')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'kl_curves_v4_aiti.png')
    plt.close(fig)
    print(f"Saved: {OUT / 'kl_curves_v4_aiti.png'}")

# ════════════════════════════════════════════════════════
# PLOT 3: Reward curves — v5b (MOAI)
# ════════════════════════════════════════════════════════
def plot_reward_curves_v5b():
    d = load('benchmark_v5b_results.json')
    fig, ax = plt.subplots(figsize=(12, 6))

    for label, hist in d['histories'].items():
        r = np.array(hist['rewards'])
        rs = smooth(r, 9)
        x = np.arange(len(rs))
        lw = 2.5 if 'mono (ema=0.80)' in label or label == 'PPO baseline' or label == 'AITI' in label else 1.5
        ax.plot(x, rs, label=label, color=get_color(label), linewidth=2.0)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (smoothed)')
    ax.set_title('Phase 4: MOAI vs AITI vs PPO — Reward Over Training')
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'reward_curves_v5b_moai.png')
    plt.close(fig)
    print(f"Saved: {OUT / 'reward_curves_v5b_moai.png'}")

# ════════════════════════════════════════════════════════
# PLOT 4: Epsilon trajectories — v5b
# ════════════════════════════════════════════════════════
def plot_epsilon_trajectories():
    d = load('benchmark_v5b_results.json')
    fig, ax = plt.subplots(figsize=(12, 5))

    for label, hist in d['histories'].items():
        eps = hist.get('epsilon', [])
        if not eps or label == 'PPO baseline':
            continue
        eps = np.array(eps)
        ax.plot(np.arange(len(eps)), eps, label=label, color=get_color(label), linewidth=2.0)

    ax.set_xlabel('Episode')
    ax.set_ylabel('ε (importance intensity)')
    ax.set_title('Epsilon Trajectories: AITI Decays to 0, MOAI Mono Finds-and-Locks, Free MOAI Oscillates')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='upper right', fontsize=9)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / 'epsilon_trajectories.png')
    plt.close(fig)
    print(f"Saved: {OUT / 'epsilon_trajectories.png'}")

# ════════════════════════════════════════════════════════
# PLOT 5: Pareto frontier (Reward vs KL)
# ════════════════════════════════════════════════════════
def plot_pareto():
    fig, ax = plt.subplots(figsize=(10, 7))

    # Combine v4 and v5b results
    all_points = []
    for fname in ['benchmark_v4_results.json', 'benchmark_v5b_results.json']:
        d = load(fname)
        for r in d['results']:
            all_points.append(r)

    # Deduplicate by label (prefer v5b)
    seen = {}
    for p in all_points:
        seen[p['label']] = p

    # Key methods to highlight
    highlight = {
        'PPO baseline', 'AITI-Advantage (linear)',
        'MOAI-Adv mono (ema=0.80)', 'MOAI-Adv free (ema=0.80)',
        'Entropy (no decay)',
    }

    for label, p in seen.items():
        kl = abs(p['kl_q4'])
        rew = p['reward_q4']
        c = get_color(label)
        if label in highlight:
            ax.scatter(kl, rew, s=120, c=c, edgecolors='black', linewidth=1.5, zorder=5)
            # Smart label positioning
            offset = (8, 8)
            if 'free' in label.lower():
                offset = (8, -15)
            elif 'Entropy' in label:
                offset = (-10, -15)
            ax.annotate(label, (kl, rew), fontsize=8, fontweight='bold',
                       xytext=offset, textcoords='offset points')
        else:
            ax.scatter(kl, rew, s=50, c=c, alpha=0.5, zorder=3)
            ax.annotate(label, (kl, rew), fontsize=7, alpha=0.6,
                       xytext=(5, 5), textcoords='offset points')

    # Draw Pareto frontier
    pareto_labels = ['PPO baseline', 'AITI-Advantage (linear)', 'MOAI-Adv mono (ema=0.80)']
    pareto_pts = [(abs(seen[l]['kl_q4']), seen[l]['reward_q4']) for l in pareto_labels if l in seen]
    pareto_pts.sort()
    if pareto_pts:
        px, py = zip(*pareto_pts)
        ax.plot(px, py, 'k--', alpha=0.4, linewidth=1.5, label='Pareto frontier')

    ax.set_xlabel('|KL Divergence| (Q4)', fontsize=12)
    ax.set_ylabel('Reward (Q4)', fontsize=12)
    ax.set_title('Pareto Frontier: Reward vs KL Divergence', fontsize=14)
    ax.legend(loc='lower right')

    # Arrow showing "ideal direction"
    ax.annotate('', xy=(0.01, 0.90), xytext=(0.06, 0.78),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(0.01, 0.91, 'Ideal\n(high reward,\nlow KL)', fontsize=8, color='green', ha='center')

    fig.tight_layout()
    fig.savefig(OUT / 'pareto_frontier.png')
    plt.close(fig)
    print(f"Saved: {OUT / 'pareto_frontier.png'}")

# ════════════════════════════════════════════════════════
# PLOT 6: Degradation — fixed importance early win, late collapse
# ════════════════════════════════════════════════════════
def plot_degradation():
    d = load('benchmark_v4_results.json')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    methods = ['PPO baseline', 'Entropy (no decay)', 'AITI-Advantage (linear)']

    for label in methods:
        hist = d['histories'].get(label, {})
        if not hist:
            continue
        r = np.array(hist['rewards'])
        rs = smooth(r, 9)
        x = np.arange(len(rs))
        ax1.plot(x, rs, label=label, color=get_color(label), linewidth=2.5)

        kl = np.array(hist['kl'])
        kls = smooth(kl, 9)
        ax2.plot(np.arange(len(kls)), kls, label=label, color=get_color(label), linewidth=2.5)

    # Shade the two regions
    for ax in [ax1, ax2]:
        ax.axvspan(0, 93, alpha=0.05, color='green', label='Importance helps')  # 100-7 for smoothing
        ax.axvspan(93, 143, alpha=0.05, color='red', label='Bias dominates')
        ax.axvline(x=93, color='red', linestyle='--', alpha=0.5)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward: Fixed Entropy Degrades, AITI Doesn\'t')
    ax1.legend(fontsize=9)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('KL Divergence')
    ax2.set_title('KL: Entropy Goes Negative, AITI Stays Low')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax2.legend(fontsize=9)

    fig.suptitle('The Bias-Variance Crossover: Why Fixed Importance Degrades', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'degradation_bias_variance.png')
    plt.close(fig)
    print(f"Saved: {OUT / 'degradation_bias_variance.png'}")

# ════════════════════════════════════════════════════════
# PLOT 7: MOAI statistics (C, rho, tau2) over training
# ════════════════════════════════════════════════════════
def plot_moai_stats():
    d = load('benchmark_v5b_results.json')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    moai_methods = [l for l in d['histories'] if 'MOAI' in l and 'mono' in l]

    for label in moai_methods:
        hist = d['histories'][label]
        c = get_color(label)

        for ax, key, title in zip(
            axes.flat,
            ['epsilon', 'moai_C', 'moai_rho', 'moai_tau2'],
            ['ε (intensity)', 'C = Cov(s, f)', 'ρ = Cov(f, (s-1)·f)', 'τ² = Var((s-1)·f)']
        ):
            vals = hist.get(key, [])
            if vals:
                ax.plot(np.arange(len(vals)), vals, label=label, color=c, linewidth=1.8)
                ax.set_title(title)
                ax.set_xlabel('Episode')

    for ax in axes.flat:
        ax.legend(fontsize=8)

    fig.suptitle('MOAI Online Statistics Over Training', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'moai_statistics.png')
    plt.close(fig)
    print(f"Saved: {OUT / 'moai_statistics.png'}")

# ════════════════════════════════════════════════════════
# PLOT 8: Bar chart — final Q4 results comparison
# ════════════════════════════════════════════════════════
def plot_bar_comparison():
    d = load('benchmark_v5b_results.json')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    labels = [r['label'] for r in d['results']]
    rewards = [r['reward_q4'] for r in d['results']]
    kls = [abs(r['kl_q4']) for r in d['results']]
    colors = [get_color(l) for l in labels]

    # Sort by reward
    order = np.argsort(rewards)[::-1]
    labels_s = [labels[i] for i in order]
    rewards_s = [rewards[i] for i in order]
    kls_s = [kls[i] for i in order]
    colors_s = [colors[i] for i in order]

    y = np.arange(len(labels_s))

    ax1.barh(y, rewards_s, color=colors_s, edgecolor='white', height=0.7)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels_s, fontsize=9)
    ax1.set_xlabel('Reward (Q4)')
    ax1.set_title('Final Reward (higher = better)')
    ax1.invert_yaxis()
    for i, v in enumerate(rewards_s):
        ax1.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=9)

    ax2.barh(y, kls_s, color=colors_s, edgecolor='white', height=0.7)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels_s, fontsize=9)
    ax2.set_xlabel('|KL Divergence| (Q4)')
    ax2.set_title('KL Divergence (lower = better)')
    ax2.invert_yaxis()
    for i, v in enumerate(kls_s):
        ax2.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)

    fig.suptitle('v5b Benchmark: Final Performance Comparison', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'bar_comparison_v5b.png')
    plt.close(fig)
    print(f"Saved: {OUT / 'bar_comparison_v5b.png'}")

# ════════════════════════════════════════════════════════
# Run all
# ════════════════════════════════════════════════════════
if __name__ == '__main__':
    plot_reward_curves_v4()
    plot_kl_curves_v4()
    plot_reward_curves_v5b()
    plot_epsilon_trajectories()
    plot_pareto()
    plot_degradation()
    plot_moai_stats()
    plot_bar_comparison()
    print(f"\nAll plots saved to {OUT}/")
