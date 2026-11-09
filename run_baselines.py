"""Stage 2: score the simple forecasters on validation and test, and draw the picture.

    python run_baselines.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines import ALL
from config import ROOT
from evaluate import HORIZONS, horizon_table, load_daily, periods


def main():
    s = load_daily()
    p = periods(s)
    print(f"{len(s)} days | train {len(p['train'])} | validation {len(p['validation'])} | test {len(p['test'])}")
    print(f"mean energy {s.mean():.1f} kWh/day, day-to-day std {s.std():.1f}\n")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, split in zip(axes, ["validation", "test"]):
        t = horizon_table(s, ALL, split)
        print(f"MAE in kWh/day, {split} (lower is better)")
        print(t.round(2).to_string(), "\n")
        t.to_csv(out / f"baselines_{split}.csv")
        for name in t.index:
            ax.plot(HORIZONS, t.loc[name], marker="o", label=name)
        ax.set_title(f"{split}"); ax.set_xlabel("forecast horizon (days ahead)"); ax.grid(alpha=.3)
    axes[0].set_ylabel("MAE (kWh/day)"); axes[0].legend()
    fig.tight_layout(); fig.savefig(out / "baselines.png", dpi=120)
    print(f"picture: {out / 'baselines.png'}")


if __name__ == "__main__":
    main()
