"""
top5.py — Static top-5-by-market-cap buy-and-hold, 2015-01-02 → 2025-12-31.

Strategy
--------
On 2015-01-02:
  1. Compute market cap for each candidate using raw price × split-adjusted
     shares (same math as backtest_6/mcap_leader.py).
  2. Pick the top 5 names by market cap.
  3. Buy each one with EXACTLY $20,000 (= $100,000 / 5) at that day's open.
  4. Hold forever. No rebalancing, no re-entry, no trims.

Same cost model: 0.1% commission + 0.05% slippage per side.

Reuses the cached prices (prices_raw_cache.parquet) and shares
(shares_cache.parquet) from backtest_6/, so no new yfinance calls.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

STARTING_CAPITAL = 100_000
TOP_N = 5
START_DATE = "2015-01-01"
END_DATE   = "2025-12-31"
ENTRY_DATE = "2015-01-02"
COMMISSION_PCT = 0.001
SLIPPAGE_PCT   = 0.0005
RISK_FREE_RATE = 0.045
BENCHMARK = "SPY"

CANDIDATES = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "XOM",
    "JPM", "V", "JNJ", "WMT", "UNH", "PG",
]

HERE = Path(__file__).resolve().parent


# --- Market cap math (copied from backtest_6/mcap_leader.py) ---
def get_split_factor(tk: str, index: pd.DatetimeIndex) -> pd.Series:
    splits = yf.Ticker(tk).splits
    if splits is None or len(splits) == 0:
        return pd.Series(1.0, index=index)
    splits.index = pd.to_datetime(splits.index).tz_localize(None)
    factor = pd.Series(1.0, index=index)
    for split_date, ratio in splits.items():
        mask = factor.index <= split_date
        factor.loc[mask] *= float(ratio)
    return factor


def entry_day_top_n(prices: pd.DataFrame, shares: pd.DataFrame, entry: pd.Timestamp, n: int) -> list[tuple[str, float]]:
    """Return top-N tickers by market cap on `entry`, with their mcap in $B."""
    print(f"Computing market cap for {len(CANDIDATES)} candidates on {entry.date()}...")
    caps = {}
    for tk in CANDIDATES:
        if tk not in shares.columns:
            continue
        shares_ts = shares[tk].dropna()
        if len(shares_ts) == 0:
            continue
        factor_at_shares = get_split_factor(tk, shares_ts.index)
        split_adj = shares_ts * factor_at_shares.reindex(shares_ts.index)
        # Outlier filter (same as backtest_6)
        med = split_adj.median()
        if med > 0:
            ratio = split_adj / med
            split_adj = split_adj[(ratio <= 2.0) & (ratio >= 0.5)]
        # Get shares value as-of entry date (back-filled from earliest available)
        sa_value = split_adj.reindex([entry], method="bfill").iloc[0]
        if pd.isna(sa_value):
            continue
        px_col = f"{tk}|Close"
        if px_col not in prices.columns or entry not in prices.index:
            continue
        px = prices.loc[entry, px_col]
        if pd.isna(px):
            continue
        caps[tk] = px * sa_value
    sorted_caps = sorted(caps.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\nRanking on {entry.date()} (top 10):")
    for i, (tk, mc) in enumerate(sorted_caps[:10], 1):
        print(f"  {i:>2}. {tk:<6}  ${mc/1e9:>7.1f}B")
    return [(tk, mc) for tk, mc in sorted_caps[:n]]


# --- Simulation ---
def simulate(prices: pd.DataFrame, top5: list[tuple[str, float]]) -> dict:
    dates = prices.index
    entry_idx = int(dates.searchsorted(pd.Timestamp(ENTRY_DATE)))
    n_days = len(dates)

    # Per-position adjusted price series (preserves dividends + splits)
    holdings = {}  # ticker -> dict(shares, entry_price, adj_close_series)
    cash = STARTING_CAPITAL
    alloc_each = STARTING_CAPITAL / TOP_N
    trade_log = []

    for tk, _ in top5:
        raw_open  = prices[f"{tk}|Open"].values
        raw_close = prices[f"{tk}|Close"].values
        adj_close = prices[f"{tk}|Adj Close"].values
        # adjusted open = raw_open * adj_close / raw_close (preserves split+div adjustment)
        adj_open = raw_open * (adj_close / raw_close)

        exec_price = adj_open[entry_idx] * (1 + SLIPPAGE_PCT)
        per_share_cost = exec_price * (1 + COMMISSION_PCT)
        shares = int(math.floor(alloc_each / per_share_cost))
        cost = shares * per_share_cost
        cash -= cost

        holdings[tk] = dict(
            shares=shares,
            entry_price=exec_price,
            adj_close=adj_close,
        )
        trade_log.append(dict(
            date=dates[entry_idx].strftime("%Y-%m-%d"),
            ticker=tk, action="buy",
            shares=shares, exec_price=round(exec_price, 4),
            gross=round(shares * exec_price, 2),
            commission=round(shares * exec_price * COMMISSION_PCT, 2),
            cash_after=round(cash, 2),
            target_alloc=round(alloc_each, 2),
            actual_alloc=round(cost, 2),
        ))

    # SPY benchmark
    spy_raw_open  = prices[f"{BENCHMARK}|Open"].values
    spy_raw_close = prices[f"{BENCHMARK}|Close"].values
    spy_adj_close = prices[f"{BENCHMARK}|Adj Close"].values
    spy_adj_open  = spy_raw_open * (spy_adj_close / spy_raw_close)
    spy_exec = spy_adj_open[entry_idx] * (1 + SLIPPAGE_PCT)
    spy_per_share = spy_exec * (1 + COMMISSION_PCT)
    spy_shares = int(math.floor(STARTING_CAPITAL / spy_per_share))
    spy_cash = STARTING_CAPITAL - spy_shares * spy_per_share

    # Mark to market every day
    strat_eq = np.full(n_days, STARTING_CAPITAL, dtype=np.float64)
    spy_eq   = np.full(n_days, STARTING_CAPITAL, dtype=np.float64)

    for i in range(n_days):
        if i < entry_idx:
            continue
        total = cash
        for tk, h in holdings.items():
            c = h["adj_close"][i]
            if not np.isnan(c):
                total += h["shares"] * c
        strat_eq[i] = total
        sc = spy_adj_close[i]
        if not np.isnan(sc):
            spy_eq[i] = spy_shares * sc + spy_cash

    # Per-position contribution to final value
    contributions = []
    for tk, h in holdings.items():
        final_px = h["adj_close"][-1]
        end_value = h["shares"] * final_px if not np.isnan(final_px) else float("nan")
        entry_cost = h["shares"] * h["entry_price"] * (1 + COMMISSION_PCT)
        gain = end_value - entry_cost
        ret_pct = end_value / entry_cost - 1 if entry_cost > 0 else float("nan")
        contributions.append(dict(
            ticker=tk,
            shares=h["shares"],
            entry_price=round(h["entry_price"], 2),
            final_price=round(final_px, 2),
            cost=round(entry_cost, 2),
            end_value=round(end_value, 2),
            gain=round(gain, 2),
            return_pct=round(ret_pct * 100, 2),
        ))

    return dict(
        trade_log=trade_log,
        strat_eq=pd.Series(strat_eq, index=dates, name="Strategy"),
        spy_eq=pd.Series(spy_eq,     index=dates, name="SPY"),
        cash_residual=cash,
        contributions=contributions,
    )


# --- Metrics ---
def metrics(eq: pd.Series, label: str) -> dict:
    eq = eq.dropna()
    rets = eq.pct_change().dropna()
    years = len(eq) / 252.0
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else float("nan")
    excess = rets - RISK_FREE_RATE / 252
    sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else float("nan")
    downside = excess[excess < 0]
    sortino = (excess.mean() / downside.std()) * np.sqrt(252) \
              if len(downside) > 0 and downside.std() > 0 else float("nan")
    mdd = (eq / eq.cummax() - 1).min()
    return dict(label=label, total_return=total, cagr=cagr,
                sharpe=sharpe, sortino=sortino, max_dd=mdd)


def main():
    prices = pd.read_parquet(HERE / "prices_raw_cache.parquet")
    shares = pd.read_parquet(HERE / "shares_cache.parquet")
    shares.index = pd.to_datetime(shares.index).tz_localize(None)

    entry_ts = pd.Timestamp(ENTRY_DATE)
    if entry_ts not in prices.index:
        # bump to next trading day
        entry_ts = prices.index[prices.index.searchsorted(entry_ts)]

    top5 = entry_day_top_n(prices, shares, entry_ts, TOP_N)
    print(f"\nTop {TOP_N} on {entry_ts.date()}:")
    for tk, mc in top5:
        print(f"  {tk:<6}  ${mc/1e9:.1f}B")

    result = simulate(prices, top5)

    # Outputs
    pd.DataFrame(result["trade_log"]).to_csv(HERE / "trade_log_top5.csv", index=False)
    print(f"\nWrote trade_log_top5.csv ({len(result['trade_log'])} rows)")

    eq_df = pd.DataFrame({"strategy": result["strat_eq"], "spy": result["spy_eq"]})
    eq_df.index.name = "date"
    eq_df.to_csv(HERE / "equity_curve_top5.csv")
    print(f"Wrote equity_curve_top5.csv")

    contrib_df = pd.DataFrame(result["contributions"])
    contrib_df.to_csv(HERE / "contributions_top5.csv", index=False)
    print(f"Wrote contributions_top5.csv")

    # Charts
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(result["strat_eq"].index, result["strat_eq"].values, label="Top-5 buy & hold", linewidth=1.5)
    ax.plot(result["spy_eq"].index,   result["spy_eq"].values,   label="SPY buy & hold",   linewidth=1.5, alpha=0.8)
    ax.set_title(f"Equity Curve: Top-{TOP_N} Static Buy-and-Hold vs SPY (entry {ENTRY_DATE})")
    ax.set_xlabel("Date"); ax.set_ylabel("Portfolio Value (USD)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / "equity_curve_top5.png", dpi=120); plt.close(fig)
    print("Wrote equity_curve_top5.png")

    dd = result["strat_eq"] / result["strat_eq"].cummax() - 1
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    ax.set_title("Top-5 Strategy Drawdown")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(HERE / "drawdown_top5.png", dpi=120); plt.close(fig)
    print("Wrote drawdown_top5.png")

    sm = metrics(result["strat_eq"], "Strategy")
    pm = metrics(result["spy_eq"], "SPY")

    def fmt_pct(x): return "n/a" if pd.isna(x) else f"{x*100:>9.2f}%"
    def fmt_num(x): return "n/a" if pd.isna(x) else f"{x:>10.3f}"

    lines = []
    lines.append("=" * 72)
    lines.append(f"  Top-{TOP_N} S&P 500 Static Buy-and-Hold — 2015-01-02 -> 2025-12-31")
    lines.append("=" * 72)
    lines.append(f"  Top-5 picked on {entry_ts.date()}: {', '.join(tk for tk, _ in top5)}")
    lines.append(f"  Equal-weight allocation: ${STARTING_CAPITAL/TOP_N:,.0f} per name")
    lines.append("=" * 72)
    lines.append(f"{'Metric':<28}{'Strategy':>20}{'SPY':>20}")
    lines.append("-" * 72)
    lines.append(f"{'Total Return':<28}{fmt_pct(sm['total_return']):>20}{fmt_pct(pm['total_return']):>20}")
    lines.append(f"{'CAGR':<28}{fmt_pct(sm['cagr']):>20}{fmt_pct(pm['cagr']):>20}")
    lines.append(f"{'Sharpe (rf=4.5%)':<28}{fmt_num(sm['sharpe']):>20}{fmt_num(pm['sharpe']):>20}")
    lines.append(f"{'Sortino':<28}{fmt_num(sm['sortino']):>20}{fmt_num(pm['sortino']):>20}")
    lines.append(f"{'Max Drawdown':<28}{fmt_pct(sm['max_dd']):>20}{fmt_pct(pm['max_dd']):>20}")
    lines.append("-" * 72)
    lines.append(f"Cash residual after buys     : ${result['cash_residual']:,.2f}")
    lines.append("")
    lines.append("Per-position contribution at end:")
    lines.append(f"{'Ticker':<8}{'Shares':>8}{'Entry $':>10}{'Final $':>10}{'Cost':>14}{'End Value':>14}{'Return':>10}")
    for c in result["contributions"]:
        lines.append(f"{c['ticker']:<8}{c['shares']:>8,}{c['entry_price']:>10.2f}{c['final_price']:>10.2f}"
                     f"{c['cost']:>14,.2f}{c['end_value']:>14,.2f}{c['return_pct']:>9.1f}%")
    lines.append("=" * 72)

    body = "\n".join(lines)
    (HERE / "metrics_summary_top5.txt").write_text(body, encoding="utf-8")
    print("\n" + body + "\n")
    print("Wrote metrics_summary_top5.txt")


if __name__ == "__main__":
    main()
