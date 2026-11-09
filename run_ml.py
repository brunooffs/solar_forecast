"""Stage 3: train the history-only ML models and compare them with the baselines.

    python run_ml.py            # trains on the training years, scores on VALIDATION (2022-2023)
    python run_ml.py --test     # once you have chosen a model: retrains on train+validation, scores on TEST

Look at --test only at the very end. If you tune after peeking at it, the number stops meaning anything.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from baselines import ALL
from config import ROOT
from evaluate import HORIZONS, VAL_END, horizon_table, load_daily
from ml_models import MODELS, MLForecaster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="final scoring on the test years (do this once)")
    a = ap.parse_args()
    s = load_daily()
    split = "test" if a.test else "validation"
    fit_end = VAL_END if a.test else None          # None -> train years only
    ml = {n: (MLForecaster(n, fit_end) if fit_end else MLForecaster(n)) for n in MODELS}
    keep = {k: ALL[k] for k in ("climatology", "anomaly_persistence")}
    table = horizon_table(s, {**keep, **ml}, split)
    mean = table.mean(axis=1).rename("mean")
    print(f"MAE in kWh/day on {split.upper()} (lower is better)"
          + (" - models retrained on train+validation" if a.test else ""))
    print(pd.concat([table.round(2), mean.round(2)], axis=1).to_string())

    clim, ap_ = table.loc["climatology"], table.loc["anomaly_persistence"]
    print("\nImprovement over climatology (positive = better, in %):")
    gain = (1 - table / clim) * 100
    print(gain.drop(index="climatology").round(1).to_string())
    print("\nBest model by mean MAE:", mean.drop(index=["climatology"]).idxmin())

    if not a.test:
        for name in ("random_forest", "gradient_boosting"):
            for h in (1, 7):
                imp = ml[name].importances(h)
                if imp is not None:
                    print(f"\n{name}, horizon {h}: most used inputs")
                    print(imp.head(5).round(3).to_string())
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    table.to_csv(out / f"ml_{split}.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in table.index:
        ax.plot(HORIZONS, table.loc[name], marker="o", label=name, ls="--" if name in ALL else "-")
    ax.set_xlabel("forecast horizon (days ahead)"); ax.set_ylabel("MAE (kWh/day)")
    ax.set_title(f"history-only models, {split}"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / f"ml_{split}.png", dpi=120)
    print(f"\npicture: {out / f'ml_{split}.png'}")


if __name__ == "__main__":
    main()
