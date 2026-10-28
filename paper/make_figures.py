"""Draw the paper's three figures from the committed result files.

    python -I paper/make_figures.py      # writes paper/figures/fig{1,2,3}_*.{pdf,png}
"""

import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"

INK = "#1f2328"
MUTED = "#6e7781"
GRID = "#d0d7de"
DRIFTED = "#c2410c"
RELEASED = "#1d4ed8"
HELPED = "#15803d"
HARMED = "#b91c1c"
REGIME = "#f3f4f6"
LAST_TRAINED = date(2023, 11, 26)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def load(name):
    return json.loads((RESULTS / name).read_text())


def days(session):
    y, m, d = (int(x) for x in session.split(".")[1:])
    return (date(y, m, d) - LAST_TRAINED).days


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def drift_curve():
    early = load("step2_per_early.json")["sessions"]
    released = {r["session"]: r for r in load("step2_per_released.json")["sessions"]}
    x = [days(r["session"]) for r in early]
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    ax.axvspan(0, 33, color=REGIME, zorder=0)
    ax.axvspan(91, max(x) + 15, color=REGIME, zorder=0)
    ax.text(16.5, 92, "near", ha="center", color=MUTED)
    ax.text((91 + max(x)) / 2, 92, "far", ha="center", color=MUTED)
    ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.plot(x, [100 * released[r["session"]]["per"] for r in early], "o-", color=RELEASED, ms=3, lw=1,
            label="released checkpoint (input layer per session)")
    ax.plot(x, [100 * r["per"] for r in early], "o-", color=DRIFTED, ms=3, lw=1,
            label="trained on sessions up to 2023.11.26 (no recalibration)")
    ax.set_xlabel("days since the last training session (2023.11.26)")
    ax.set_ylabel("phoneme error rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(min(x) - 10, max(x) + 15)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    save(fig, "fig2_drift")


def changes_panel(ax, title, arms):
    labels, helped, harmed = [], [], []
    for name, arm in arms:
        for policy, tag in (("always", "always"), ("logistic_gate", "gate")):
            labels.append(f"{name}\n{tag}")
            helped.append(arm["applied"][policy]["helped"])
            harmed.append(arm["applied"][policy]["harmed"])
    pos = list(range(len(labels)))
    ax.bar([p - 0.2 for p in pos], helped, 0.4, color=HELPED, label="helped")
    ax.bar([p + 0.2 for p in pos], harmed, 0.4, color=HARMED, label="harmed")
    for p, h, b in zip(pos, helped, harmed):
        ax.text(p - 0.2, h + 1, str(h), ha="center", va="bottom", fontsize=7, color=INK)
        ax.text(p + 0.2, b + 1, str(b), ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(pos, labels, fontsize=7.5)
    ax.set_title(title)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def gate_changes():
    t15 = load("step3a_gate.json")["temporal"]["arms"]
    t12 = load("step3a_gate_t12.json")["temporal"]["arms"]
    near = load("step4_rescoring.json")["R5"]["near"]["arms"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True,
                             gridspec_kw={"width_ratios": [3, 1.4, 2]})
    changes_panel(axes[0], "A  T15 calibrated",
                  [("OPT", t15["opt"]), ("Qwen", t15["qwen"]), ("Jev", t15["jev"])])
    changes_panel(axes[1], "B  T12 calibrated", [("OPT", t12["opt"])])
    changes_panel(axes[2], "C  T15 drifted, near", [("OPT", near["opt"]), ("Qwen", near["qwen"])])
    axes[0].set_ylabel("sentences changed")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig1_gate_changes")


def regimes():
    r5 = load("step4_rescoring.json")["R5"]
    bars = [("first pass", "none", "opt", MUTED), ("OPT always", "always", "opt", "#9a3412"),
            ("OPT gate", "logistic_gate", "opt", "#fdba74"), ("Qwen always", "always", "qwen", "#1e3a8a"),
            ("Qwen gate", "logistic_gate", "qwen", "#93c5fd"), ("OPT oracle gate", "oracle", "opt", INK)]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    oracle = load("step4_rescoring.json")["R3"]
    for ax, regime, title in ((axes[0], "near", "A  near: 5 sessions, 189 sentences"),
                              (axes[1], "far", "B  far: 15 sessions, 462 sentences")):
        arms = r5[regime]["arms"]
        best = oracle[f"{regime}_oracle_wer"]
        ax.axhline(best, color=INK, lw=0.8, ls="--")
        title += f"\nbest of 100 candidates (dashed): {best:.1f}"
        values = [arms[arm]["wer"][policy] for _, policy, arm, _ in bars]
        ax.bar(range(len(bars)), values, color=[b[3] for b in bars])
        for i, v in enumerate(values):
            ax.text(i, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(bars)), [b[0] for b in bars], rotation=35, ha="right", fontsize=7.5)
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("word error rate (%)")
    fig.tight_layout()
    save(fig, "fig3_regimes")


if __name__ == "__main__":
    drift_curve()
    gate_changes()
    regimes()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
