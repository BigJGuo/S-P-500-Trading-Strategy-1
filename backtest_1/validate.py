"""
validate.py — orchestrate smoke test, full backtest, and walk-forward folds.

Why this exists
---------------
The strategy in backtest.py has no learned parameters (the +10%/-5% thresholds
and 0.01%/0.005% sell fractions are spec-fixed), so classical overfitting is
not a concern. The genuine robustness question is:

  "Are the returns an artifact of the 2015–2024 bull market, or does the
   strategy behave consistently across different regimes?"

This script answers that by running the same strategy across 5 non-overlapping
2-year windows (folds). Wildly different per-fold results would indicate the
full-period number leans on one particular regime; consistent results across
all five would confirm the strategy is regime-agnostic.

Runs
----
1. SMOKE TEST   — 10 mega-caps over 6 months. End-to-end pipeline check; ~30s.
2. FULL         — S&P 500, 2015-01-02 → 2024-12-31. Primary result.
3. FOLD 1       — 2015-01-02 → 2016-12-31  (low-vol, modest gains)
4. FOLD 2       — 2017-01-03 → 2018-12-31  (Trump rally + Q4 2018 selloff)
5. FOLD 3       — 2019-01-02 → 2020-12-31  (incl. COVID crash + recovery)
6. FOLD 4       — 2021-01-04 → 2022-12-31  (incl. 2022 bear market)
7. FOLD 5       — 2023-01-03 → 2024-12-31  (AI rally / new highs)

Then aggregates all metrics_summary*.csv files into a single
validation_summary.txt + folds_comparison.csv table for easy review.

Usage
-----
    python validate.py             # full pipeline (~10-15 min first run, cache reuse after)
    python validate.py --skip-smoke
    python validate.py --skip-full   (run folds only — assumes cache exists)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE   = Path(__file__).resolve().parent
SCRIPT = HERE / "backtest.py"
PY     = sys.executable

FOLDS = [
    # (start,       end,          entry,        suffix)
    ("2015-01-01", "2016-12-31", "2015-01-02", "_fold1_2015_2016"),
    ("2017-01-01", "2018-12-31", "2017-01-03", "_fold2_2017_2018"),
    ("2019-01-01", "2020-12-31", "2019-01-02", "_fold3_2019_2020"),
    ("2021-01-01", "2022-12-31", "2021-01-04", "_fold4_2021_2022"),
    ("2023-01-01", "2024-12-31", "2023-01-03", "_fold5_2023_2024"),
]


def run(args: list[str], desc: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {desc}\n{bar}", flush=True)
    subprocess.run([PY, str(SCRIPT)] + args, check=True)


def aggregate() -> None:
    """Collect every metrics_summary*.csv and write a side-by-side table."""
    rows = []

    full = HERE / "metrics_summary.csv"
    if full.exists():
        df = pd.read_csv(full)
        df.insert(0, "label", "FULL 2015-2024")
        rows.append(df)

    for _, _, _, suffix in FOLDS:
        f = HERE / f"metrics_summary{suffix}.csv"
        if f.exists():
            df = pd.read_csv(f)
            df.insert(0, "label", suffix.lstrip("_").replace("_", " "))
            rows.append(df)

    if not rows:
        print("No metric CSVs found to aggregate.")
        return

    combined = pd.concat(rows, ignore_index=True)
    out_csv = HERE / "folds_comparison.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv.name}")

    # Pretty text summary too.
    def pct(x):
        return "n/a" if pd.isna(x) else f"{x*100:>8.2f}%"
    def num(x):
        return "n/a" if pd.isna(x) else f"{x:>8.3f}"

    lines = []
    lines.append("=" * 110)
    lines.append("  WALK-FORWARD VALIDATION — Strategy vs SPY across non-overlapping windows")
    lines.append("=" * 110)
    header = (
        f"{'Period':<22}"
        f"{'Strat Ret':>11}{'SPY Ret':>11}"
        f"{'Strat CAGR':>12}{'SPY CAGR':>11}"
        f"{'Strat Shrp':>12}{'SPY Shrp':>11}"
        f"{'Strat MDD':>11}{'SPY MDD':>11}"
    )
    lines.append(header)
    lines.append("-" * 110)
    for _, r in combined.iterrows():
        lines.append(
            f"{r['label']:<22}"
            f"{pct(r['strat_total_return']):>11}{pct(r['spy_total_return']):>11}"
            f"{pct(r['strat_cagr']):>12}{pct(r['spy_cagr']):>11}"
            f"{num(r['strat_sharpe']):>12}{num(r['spy_sharpe']):>11}"
            f"{pct(r['strat_max_dd']):>11}{pct(r['spy_max_dd']):>11}"
        )
    lines.append("-" * 110)
    lines.append("Trade-level stats per period:")
    lines.append(f"{'Period':<22}{'Trades':>9}{'PT fills':>10}{'SL fills':>10}{'Closed':>9}{'Win %':>10}{'Avg PT':>10}{'Avg SL':>10}")
    for _, r in combined.iterrows():
        lines.append(
            f"{r['label']:<22}"
            f"{int(r['total_trades']):>9,}"
            f"{int(r['profit_take_fills']):>10,}"
            f"{int(r['stop_loss_fills']):>10,}"
            f"{int(r['closed_positions']):>9,}"
            f"{pct(r['win_rate']):>10}"
            f"{pct(r['avg_gain_profit_take']):>10}"
            f"{pct(r['avg_loss_stop_loss']):>10}"
        )
    lines.append("=" * 110)
    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("  - Consistent Sharpe across folds (low cross-fold std) -> strategy is regime-agnostic.")
    lines.append("  - Strat returns tracking SPY closely in every fold    -> partial-trim is largely cosmetic; not adding alpha.")
    lines.append("  - One fold dominating the full-period number          -> result is fragile / regime-dependent.")
    lines.append("  - Win rate << 50% but positive return                 -> a few large winners carrying the strategy.")
    lines.append("")
    lines.append("Note: this strategy has NO learned parameters — thresholds are spec-fixed.")
    lines.append("Classical overfitting is impossible. Walk-forward here tests regime stability,")
    lines.append("not parameter generalization.")

    out_txt = HERE / "validation_summary.txt"
    out_txt.write_text("\n".join(lines))
    print(f"Wrote {out_txt.name}")
    print("\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-full",  action="store_true")
    parser.add_argument("--skip-folds", action="store_true")
    parser.add_argument("--refresh",    action="store_true",
                        help="Force re-download (passed to backtest.py).")
    args = parser.parse_args()

    if not args.skip_smoke:
        run(["--smoke"], "1/7  SMOKE TEST (10 tickers, 2024-H1)")

    if not args.skip_full:
        full_args = []
        if args.refresh:
            full_args.append("--refresh")
        run(full_args, "2/7  FULL BACKTEST 2015-2024")

    if not args.skip_folds:
        for i, (start, end, entry, suffix) in enumerate(FOLDS, start=3):
            run([
                "--start", start, "--end", end,
                "--entry", entry, "--suffix", suffix,
            ], f"{i}/7  FOLD {suffix.lstrip('_')}")

    aggregate()


if __name__ == "__main__":
    main()
