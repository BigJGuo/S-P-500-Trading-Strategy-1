"""
compare.py — run the three strategy variants and produce a side-by-side report.

Variants (each is one continuous backtest 2015-01-02 → 2025-12-31):
  v1_small_pct  — tiny percentages (0.01% PT / 0.005% SL), with the
                  max(1 share) floor on. Closest to the original $10/$5 spec
                  in spirit, but with the price-rounding dead-end fixed.
  v2_reinvest   — current 0.1% / 0.05% sizing, with the max(1 share) floor on,
                  but cash from trims is rolled into SPY at each day's close
                  so it doesn't drag the portfolio.
  v3_no_floor   — current 0.1% / 0.05% sizing, max(1 share) floor REMOVED.
                  Trims only fire when floor(shares * pct) ≥ 1, so the
                  strategy is silent on small positions and scales naturally
                  with position growth.

After running, prints a comparison table and writes comparison_summary.txt
plus per-variant per-period slice reports (validate.py with --suffix).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE   = Path(__file__).resolve().parent
SCRIPT = HERE / "backtest.py"
PY     = sys.executable

VARIANTS = [
    dict(
        name="v1_small_pct",
        label="v1  small %  (0.01 / 0.005, min-share floor)",
        args=["--pt-pct", "0.0001", "--sl-pct", "0.00005"],
    ),
    dict(
        name="v2_reinvest",
        label="v2  reinvest (0.1  / 0.05,  min-share floor, SPY reinvest)",
        args=["--pt-pct", "0.001",  "--sl-pct", "0.0005", "--reinvest"],
    ),
    dict(
        name="v3_no_floor",
        label="v3  no floor (0.1  / 0.05,  NO min-share floor)",
        args=["--pt-pct", "0.001",  "--sl-pct", "0.0005", "--no-min-share"],
    ),
]


def run(args: list[str], desc: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {desc}\n{bar}", flush=True)
    subprocess.run([PY, str(SCRIPT)] + args, check=True)


def trades_after(ticker_csv: Path, date: str) -> int:
    df = pd.read_csv(ticker_csv, parse_dates=["date"])
    non_entry = df[df["trigger_type"] != "entry"]
    return int((non_entry["date"] >= pd.Timestamp(date)).sum())


def main():
    # 1) Run each variant
    for v in VARIANTS:
        suffix = f"_{v['name']}"
        run(v["args"] + ["--suffix", suffix], f"RUN {v['label']}")

    # 2) Aggregate metrics CSVs
    rows = []
    for v in VARIANTS:
        csv = HERE / f"metrics_summary_{v['name']}.csv"
        if not csv.exists():
            print(f"WARN: missing {csv.name}")
            continue
        df = pd.read_csv(csv).iloc[0].to_dict()
        df["variant"] = v["name"]
        df["label"]   = v["label"]
        # additional liveness check: trades after 2020
        log = HERE / f"trade_log_{v['name']}.csv"
        df["trades_post_2020"] = trades_after(log, "2021-01-01") if log.exists() else -1
        rows.append(df)

    out_csv = HERE / "comparison_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv.name}")

    # 3) Pretty text report
    def pct(x): return "n/a" if pd.isna(x) else f"{x*100:>8.2f}%"
    def num(x): return "n/a" if pd.isna(x) else f"{x:>8.3f}"

    lines = []
    lines.append("=" * 110)
    lines.append("  STRATEGY VARIANT COMPARISON — 2015-01-02 -> 2025-12-31, one continuous portfolio per variant")
    lines.append("=" * 110)
    lines.append("")
    lines.append(f"{'Variant':<55}{'Strat Ret':>11}{'SPY Ret':>11}{'Strat CAGR':>12}{'Strat Shrp':>12}{'Strat MDD':>11}")
    lines.append("-" * 110)
    for r in rows:
        lines.append(
            f"{r['label']:<55}"
            f"{pct(r['strat_total_return']):>11}{pct(r['spy_total_return']):>11}"
            f"{pct(r['strat_cagr']):>12}{num(r['strat_sharpe']):>12}{pct(r['strat_max_dd']):>11}"
        )
    lines.append("")
    lines.append(f"{'Variant':<55}{'Trades':>10}{'PT fills':>10}{'SL fills':>10}{'Closed':>9}{'Post-2020':>11}")
    lines.append("-" * 110)
    for r in rows:
        lines.append(
            f"{r['label']:<55}"
            f"{int(r['total_trades']):>10,}"
            f"{int(r['profit_take_fills']):>10,}"
            f"{int(r['stop_loss_fills']):>10,}"
            f"{int(r['closed_positions']):>9,}"
            f"{int(r['trades_post_2020']):>11,}"
        )
    lines.append("")
    lines.append("=" * 110)
    lines.append("")
    lines.append("Reading guide:")
    lines.append("  v1 — gentlest. Confirms the % sizing fixes the post-2020 inactivity even at very small %.")
    lines.append("  v2 — same trim aggressiveness as backtest_3/4 but cash → SPY. Strategy benefits from the")
    lines.append("       trimmed names AND from market exposure on idle cash, so it should track SPY closely.")
    lines.append("  v3 — small positions never trim. By 2025 most positions are large enough for floor(shares*%)")
    lines.append("       to fill ≥1, so trims still happen — but only when economically meaningful.")
    lines.append("")
    lines.append("For prior context: backtest_2 ($10/$5 fixed) returned 326.27%, max DD -33.49%, but stopped")
    lines.append("trading entirely after 2020-07-07. backtest_3 (1%/0.5%) and backtest_4 (0.1%/0.05%) traded")
    lines.append("through 2025 but the cash drag from non-reinvested trims dropped returns to 26.92% and 49.07%.")

    out_txt = HERE / "comparison_summary.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_txt.name}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
