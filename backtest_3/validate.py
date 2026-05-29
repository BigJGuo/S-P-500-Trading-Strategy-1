"""
validate.py — single continuous backtest 2015->2025 + sliced per-period report.

Why this exists
---------------
The strategy is *one* continuous portfolio: buy all S&P 500 names on
2015-01-02, partial-trim forever, never re-enter. The earlier multi-fold
implementation (in backtest_1/) restarted the portfolio every 2 years,
which contradicted the spec.

This rewrite runs ONE backtest from 2015-01-02 through 2025-12-31, then
**slices** the resulting equity curve into 2-year sub-periods for the
robustness report. Per-period stats describe the actual returns the live
portfolio earned during each window — they are NOT independent backtests
and the periods cannot be added together.

Runs
----
1. SMOKE TEST   — 10 mega-caps over 6 months. Pipeline sanity check.
2. CONTINUOUS   — S&P 500, 2015-01-02 -> 2025-12-31. One portfolio.
3. SLICE        — read equity_curve.csv + trade_log.csv, compute window
                  metrics, write validation_summary.txt and
                  periods_comparison.csv.

Usage
-----
    python validate.py                  # full pipeline
    python validate.py --refresh        # force fresh yfinance download
    python validate.py --skip-smoke
    python validate.py --skip-backtest  # slice only (cache must exist)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE       = Path(__file__).resolve().parent
SCRIPT     = HERE / "backtest.py"
PY         = sys.executable
EQUITY_CSV = HERE / "equity_curve.csv"
TRADE_LOG  = HERE / "trade_log.csv"

RISK_FREE_RATE = 0.045

PERIODS = [
    # (label,       start,         end)
    ("2015-2016",   "2015-01-01", "2016-12-31"),
    ("2017-2018",   "2017-01-01", "2018-12-31"),
    ("2019-2020",   "2019-01-01", "2020-12-31"),
    ("2021-2022",   "2021-01-01", "2022-12-31"),
    ("2023-2024",   "2023-01-01", "2024-12-31"),
    ("2025",        "2025-01-01", "2025-12-31"),
]


def run(args: list[str], desc: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {desc}\n{bar}", flush=True)
    subprocess.run([PY, str(SCRIPT)] + args, check=True)


def window_metrics(eq: pd.Series) -> dict:
    """Compute returns/Sharpe/Sortino/MDD over an equity slice.

    Note: total_return here is the % change of portfolio value across the
    window — i.e. what the live portfolio actually earned during those dates.
    It is NOT the result of a fresh-restart backtest over the window.
    """
    eq = eq.dropna()
    if len(eq) < 2:
        return dict(total_return=float("nan"), cagr=float("nan"),
                    sharpe=float("nan"), sortino=float("nan"), max_dd=float("nan"))
    rets = eq.pct_change().dropna()
    years = len(eq) / 252.0
    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else float("nan")

    excess = rets - RISK_FREE_RATE / 252.0
    sharpe  = (excess.mean() / excess.std())  * np.sqrt(252) if excess.std() > 0 else float("nan")
    downside = excess[excess < 0]
    sortino = (excess.mean() / downside.std()) * np.sqrt(252) \
              if len(downside) > 0 and downside.std() > 0 else float("nan")

    max_dd = (eq / eq.cummax() - 1).min()
    return dict(
        total_return=total_return, cagr=cagr,
        sharpe=sharpe, sortino=sortino, max_dd=max_dd,
    )


def aggregate() -> None:
    if not EQUITY_CSV.exists():
        print(f"Cannot slice: {EQUITY_CSV.name} not found. Run the backtest first.")
        return
    if not TRADE_LOG.exists():
        print(f"Cannot slice trades: {TRADE_LOG.name} not found.")
        return

    eq = pd.read_csv(EQUITY_CSV, parse_dates=["date"], index_col="date")
    trades = pd.read_csv(TRADE_LOG, parse_dates=["date"])

    def trade_counts(sub: pd.DataFrame) -> dict:
        return dict(
            total_trades=len(sub),
            entries=int((sub["trigger_type"] == "entry").sum()),
            profit_take_fills=int((sub["trigger_type"] == "profit_take").sum()),
            stop_loss_fills=int((sub["trigger_type"] == "stop_loss").sum()),
            full_closes=int((sub["trigger_type"] == "full_close").sum()),
        )

    rows = []
    # FULL row first
    strat_m = window_metrics(eq["strategy"])
    spy_m   = window_metrics(eq["spy"])
    tc      = trade_counts(trades)
    rows.append(dict(label="FULL 2015-2025",
                     start=str(eq.index.min().date()), end=str(eq.index.max().date()),
                     **{f"strat_{k}": v for k, v in strat_m.items()},
                     **{f"spy_{k}":   v for k, v in spy_m.items()},
                     **tc))

    for label, start, end in PERIODS:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        eq_slice = eq.loc[(eq.index >= s) & (eq.index <= e)]
        tr_slice = trades.loc[(trades["date"] >= s) & (trades["date"] <= e)]
        if len(eq_slice) < 30:
            continue
        rows.append(dict(label=label,
                         start=str(eq_slice.index.min().date()),
                         end=str(eq_slice.index.max().date()),
                         **{f"strat_{k}": v for k, v in window_metrics(eq_slice["strategy"]).items()},
                         **{f"spy_{k}":   v for k, v in window_metrics(eq_slice["spy"]).items()},
                         **trade_counts(tr_slice)))

    df = pd.DataFrame(rows)
    out_csv = HERE / "periods_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv.name}")

    # Pretty text report.
    def pct(x): return "n/a" if pd.isna(x) else f"{x*100:>8.2f}%"
    def num(x): return "n/a" if pd.isna(x) else f"{x:>8.3f}"

    lines = []
    lines.append("=" * 110)
    lines.append("  CONTINUOUS BACKTEST 2015->2025 — Per-Period Slice Report")
    lines.append("=" * 110)
    lines.append("")
    lines.append("All numbers below are from ONE continuous portfolio. The strategy was bought once")
    lines.append("on 2015-01-02 and has never re-entered. Per-period rows describe the actual returns")
    lines.append("the live portfolio earned during those dates — not independent backtests.")
    lines.append("")
    lines.append("-" * 110)
    header = (
        f"{'Period':<16}{'Days':>6}"
        f"{'Strat Ret':>11}{'SPY Ret':>11}"
        f"{'Strat CAGR':>12}{'SPY CAGR':>11}"
        f"{'Strat Shrp':>12}{'SPY Shrp':>11}"
        f"{'Strat MDD':>11}{'SPY MDD':>11}"
    )
    lines.append(header)
    lines.append("-" * 110)
    for _, r in df.iterrows():
        # rough day count from start/end strings
        days = (pd.Timestamp(r["end"]) - pd.Timestamp(r["start"])).days
        lines.append(
            f"{r['label']:<16}{days:>6}"
            f"{pct(r['strat_total_return']):>11}{pct(r['spy_total_return']):>11}"
            f"{pct(r['strat_cagr']):>12}{pct(r['spy_cagr']):>11}"
            f"{num(r['strat_sharpe']):>12}{num(r['spy_sharpe']):>11}"
            f"{pct(r['strat_max_dd']):>11}{pct(r['spy_max_dd']):>11}"
        )
    lines.append("-" * 110)
    lines.append("")
    lines.append("Trades that occurred during each window (slices of the single trade log):")
    lines.append(f"{'Period':<16}{'Total':>8}{'Entries':>10}{'PT fills':>10}{'SL fills':>10}{'Full close':>12}")
    for _, r in df.iterrows():
        lines.append(
            f"{r['label']:<16}{int(r['total_trades']):>8,}"
            f"{int(r['entries']):>10,}{int(r['profit_take_fills']):>10,}"
            f"{int(r['stop_loss_fills']):>10,}{int(r['full_closes']):>12,}"
        )
    lines.append("=" * 110)
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - Entries should appear ONLY in the 2015-2016 row (and the FULL row). If 'Entries' is")
    lines.append("    non-zero for any later period, the strategy re-entered — that would be a bug.")
    lines.append("  - Consistent Sharpe across periods -> strategy is regime-agnostic.")
    lines.append("  - One period dominating the full return -> result leans on that regime.")
    lines.append("  - This strategy has NO learned parameters (thresholds are spec-fixed),")
    lines.append("    so classical overfitting is impossible. Per-period slicing tests regime stability.")

    out_txt = HERE / "validation_summary.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_txt.name}")
    print("\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-smoke",    action="store_true")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Skip the continuous backtest; just slice an existing equity_curve.csv.")
    parser.add_argument("--refresh",       action="store_true",
                        help="Force re-download of price data (passed to backtest.py).")
    args = parser.parse_args()

    if not args.skip_smoke:
        run(["--smoke"], "1/2  SMOKE TEST (10 tickers, 2024-H1)")

    if not args.skip_backtest:
        bt_args = []
        if args.refresh:
            bt_args.append("--refresh")
        run(bt_args, "2/2  CONTINUOUS BACKTEST 2015-2025 (one portfolio)")

    aggregate()


if __name__ == "__main__":
    main()
