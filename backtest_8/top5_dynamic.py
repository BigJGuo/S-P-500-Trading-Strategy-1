"""
top5_dynamic.py — Dynamic top-5-by-market-cap holdings with 20% hysteresis.

Strategy
--------
On 2015-01-02: buy the current top-5 by market cap, equal-weight ($20k each).

Daily after that:
  1. Rank ALL candidates by current market cap (raw price × split-adj shares).
  2. Identify the holding with the LOWEST current market cap among our 5
     positions — call this `worst_holding`.
  3. Identify the highest-mcap company NOT in our holdings — call this
     `top_challenger`.
  4. If mcap(top_challenger) > 1.20 × mcap(worst_holding), queue a swap:
     sell ALL of worst_holding at next day's open and use the proceeds to
     buy as much of top_challenger as possible. Repeat the check (with
     updated holdings) in case another swap is warranted same day.
  5. Otherwise, do nothing.

The 1.20× threshold is the hysteresis the user asked for — it prevents
flipping when two companies are close in size. A holding only gets kicked
out when the challenger is meaningfully larger.

Note: only the displaced holding's shares are rolled into the new name.
The other 4 positions are left alone (no equal-weight rebalancing on every
swap, which would add a lot of churn). Over time positions drift away
from equal weight, but that matches the spirit of "minimal trading."

Cost model: 0.1% commission + 0.05% slippage per side, same as before.

Reuses prices_raw_cache.parquet and shares_cache.parquet from backtest_6/.
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

# --- Configuration ---
STARTING_CAPITAL = 100_000
TOP_N = 5
SWAP_THRESHOLD = 1.20         # challenger must be >= 20% bigger to swap
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


# --- Market cap math (matching backtest_6/mcap_leader.py) ---
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


def daily_market_cap(prices: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    raw_close = pd.DataFrame({tk: prices[f"{tk}|Close"] for tk in CANDIDATES})
    print("Computing split-adjusted shares...")
    mcap = pd.DataFrame(index=raw_close.index, columns=CANDIDATES, dtype="float64")
    for tk in CANDIDATES:
        if tk not in shares.columns:
            continue
        shares_ts = shares[tk].dropna()
        if len(shares_ts) == 0:
            continue
        factor_at_shares = get_split_factor(tk, shares_ts.index)
        split_adj = shares_ts * factor_at_shares.reindex(shares_ts.index)
        med = split_adj.median()
        if med > 0:
            ratio = split_adj / med
            split_adj = split_adj[(ratio <= 2.0) & (ratio >= 0.5)]
        sa_daily = split_adj.reindex(raw_close.index).ffill().bfill()
        mcap[tk] = raw_close[tk] * sa_daily
    return mcap


# --- Simulation ---
def simulate(prices: pd.DataFrame, mcap: pd.DataFrame) -> dict:
    dates = mcap.index
    entry_idx = int(dates.searchsorted(pd.Timestamp(ENTRY_DATE)))
    n_days = len(dates)

    # Pre-extract series as numpy
    raw_close = {tk: prices[f"{tk}|Close"].values     for tk in CANDIDATES}
    raw_open  = {tk: prices[f"{tk}|Open"].values      for tk in CANDIDATES}
    adj_close = {tk: prices[f"{tk}|Adj Close"].values for tk in CANDIDATES}
    adj_open  = {tk: (prices[f"{tk}|Open"] * prices[f"{tk}|Adj Close"] / prices[f"{tk}|Close"]).values
                 for tk in CANDIDATES}

    # Pick initial top 5 by mcap on entry day
    entry_caps = mcap.iloc[entry_idx].dropna().sort_values(ascending=False)
    initial = list(entry_caps.head(TOP_N).index)
    print(f"\nInitial top-{TOP_N} on {dates[entry_idx].date()}:")
    for tk in initial:
        print(f"  {tk:<6}  ${entry_caps[tk]/1e9:>7.1f}B")

    # holdings[ticker] = shares
    holdings: dict[str, int] = {}
    cash = STARTING_CAPITAL
    alloc_each = STARTING_CAPITAL / TOP_N
    trade_log = []

    for tk in initial:
        exec_price = adj_open[tk][entry_idx] * (1 + SLIPPAGE_PCT)
        per_share_cost = exec_price * (1 + COMMISSION_PCT)
        shares = int(math.floor(alloc_each / per_share_cost))
        cost = shares * per_share_cost
        cash -= cost
        holdings[tk] = shares
        trade_log.append(dict(
            date=dates[entry_idx].strftime("%Y-%m-%d"),
            ticker=tk, action="buy",
            shares=shares, exec_price=round(exec_price, 4),
            gross=round(shares * exec_price, 2),
            commission=round(shares * exec_price * COMMISSION_PCT, 2),
            cash_after=round(cash, 2),
            reason="initial_entry",
            mcap_billions=round(float(entry_caps[tk]) / 1e9, 1),
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

    strat_eq = np.full(n_days, STARTING_CAPITAL, dtype=np.float64)
    spy_eq   = np.full(n_days, STARTING_CAPITAL, dtype=np.float64)

    # Daily pending: list of (action_type, ticker_to_sell, ticker_to_buy)
    pending_swaps: list[tuple[str, str]] = []
    composition_history = []  # (date, set_of_tickers, ranking_info)

    for i in range(n_days):
        if i < entry_idx:
            continue

        # 1) Execute any pending swaps at TODAY's open
        for ticker_out, ticker_in in pending_swaps:
            if ticker_out not in holdings:
                continue  # already sold via earlier swap (shouldn't happen)
            sell_open = adj_open[ticker_out][i]
            buy_open  = adj_open[ticker_in][i]
            if np.isnan(sell_open) or np.isnan(buy_open):
                continue  # no fill possible today, drop the swap
            # Sell all of ticker_out
            sell_price = sell_open * (1 - SLIPPAGE_PCT)
            sell_shares = holdings[ticker_out]
            sell_gross = sell_shares * sell_price
            sell_comm  = sell_gross * COMMISSION_PCT
            proceeds   = sell_gross - sell_comm
            cash += proceeds
            trade_log.append(dict(
                date=dates[i].strftime("%Y-%m-%d"),
                ticker=ticker_out, action="sell",
                shares=sell_shares, exec_price=round(sell_price, 4),
                gross=round(sell_gross, 2),
                commission=round(sell_comm, 2),
                cash_after=round(cash, 2),
                reason=f"replaced_by_{ticker_in}",
                mcap_billions=round(float(mcap.iloc[i-1].get(ticker_out, float("nan"))) / 1e9, 1),
            ))
            del holdings[ticker_out]
            # Buy ticker_in with the proceeds
            buy_price = buy_open * (1 + SLIPPAGE_PCT)
            per_share_cost = buy_price * (1 + COMMISSION_PCT)
            new_shares = int(math.floor(cash / per_share_cost)) if per_share_cost > 0 else 0
            # Don't drain ALL cash — only the proceeds-equivalent. Leave the
            # original $182 startup residual in place. Simplest: use proceeds.
            # Actually: per_share_cost may overshoot if cash >> proceeds; rein in.
            target_cash = proceeds
            new_shares = int(math.floor(target_cash / per_share_cost)) if per_share_cost > 0 else 0
            cost = new_shares * per_share_cost
            cash -= cost
            holdings[ticker_in] = new_shares
            trade_log.append(dict(
                date=dates[i].strftime("%Y-%m-%d"),
                ticker=ticker_in, action="buy",
                shares=new_shares, exec_price=round(buy_price, 4),
                gross=round(new_shares * buy_price, 2),
                commission=round(new_shares * buy_price * COMMISSION_PCT, 2),
                cash_after=round(cash, 2),
                reason=f"replaces_{ticker_out}",
                mcap_billions=round(float(mcap.iloc[i-1].get(ticker_in, float("nan"))) / 1e9, 1),
            ))
        pending_swaps = []

        # 2) Mark to market at today's close
        total = cash
        for tk, sh in holdings.items():
            c = adj_close[tk][i]
            if not np.isnan(c):
                total += sh * c
        strat_eq[i] = total
        sc = spy_adj_close[i]
        if not np.isnan(sc):
            spy_eq[i] = spy_shares * sc + spy_cash

        # 3) Evaluate swap candidates at today's close
        # Iteratively: lowest-mcap holding vs highest-mcap non-holding
        # If non-holding is > 1.20 × holding, queue a swap. Repeat until stable.
        today_caps = mcap.iloc[i]
        sim_holdings = dict(holdings)
        local_swaps = []
        while True:
            # mcap of each current holding
            hold_caps = {tk: today_caps.get(tk, float("nan")) for tk in sim_holdings}
            hold_caps = {tk: v for tk, v in hold_caps.items() if pd.notna(v) and v > 0}
            if not hold_caps:
                break
            worst_tk = min(hold_caps, key=hold_caps.get)
            worst_mcap = hold_caps[worst_tk]
            # non-holding candidates
            non_held = {tk: today_caps.get(tk, float("nan")) for tk in CANDIDATES if tk not in sim_holdings}
            non_held = {tk: v for tk, v in non_held.items() if pd.notna(v) and v > 0}
            if not non_held:
                break
            best_challenger = max(non_held, key=non_held.get)
            challenger_mcap = non_held[best_challenger]
            if challenger_mcap > SWAP_THRESHOLD * worst_mcap:
                local_swaps.append((worst_tk, best_challenger))
                # apply to sim_holdings to detect further swaps
                del sim_holdings[worst_tk]
                sim_holdings[best_challenger] = 0  # placeholder
            else:
                break
        if local_swaps:
            pending_swaps.extend(local_swaps)

        composition_history.append((dates[i], tuple(sorted(holdings.keys()))))

    # Per-position contribution at end
    contributions = []
    for tk, sh in holdings.items():
        final_px = adj_close[tk][-1]
        end_value = sh * final_px if not np.isnan(final_px) else float("nan")
        contributions.append(dict(ticker=tk, shares=sh,
                                  final_price=round(float(final_px), 2),
                                  end_value=round(float(end_value), 2)))

    # Composition timeline (just transitions)
    timeline = []
    prev = None
    for d, comp in composition_history:
        if comp != prev:
            timeline.append((d, set(comp)))
            prev = comp

    return dict(
        trade_log=trade_log,
        strat_eq=pd.Series(strat_eq, index=dates, name="Strategy"),
        spy_eq=pd.Series(spy_eq,     index=dates, name="SPY"),
        final_holdings=dict(holdings),
        final_cash=cash,
        contributions=contributions,
        timeline=timeline,
    )


# --- Metrics ---
def metrics(eq: pd.Series) -> dict:
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
    return dict(total_return=total, cagr=cagr, sharpe=sharpe, sortino=sortino, max_dd=mdd)


def main():
    prices = pd.read_parquet(HERE / "prices_raw_cache.parquet")
    shares = pd.read_parquet(HERE / "shares_cache.parquet")
    shares.index = pd.to_datetime(shares.index).tz_localize(None)

    mcap = daily_market_cap(prices, shares)
    result = simulate(prices, mcap)

    # Write outputs
    pd.DataFrame(result["trade_log"]).to_csv(HERE / "trade_log_top5_dynamic.csv", index=False)
    print(f"\nWrote trade_log_top5_dynamic.csv ({len(result['trade_log'])} rows)")

    eq_df = pd.DataFrame({"strategy": result["strat_eq"], "spy": result["spy_eq"]})
    eq_df.index.name = "date"
    eq_df.to_csv(HERE / "equity_curve_top5_dynamic.csv")

    # Composition timeline
    tl = pd.DataFrame([(d.date(), sorted(s)) for d, s in result["timeline"]],
                      columns=["date", "holdings"])
    tl.to_csv(HERE / "composition_timeline.csv", index=False)
    print(f"Wrote composition_timeline.csv ({len(tl)} composition changes)")

    pd.DataFrame(result["contributions"]).to_csv(HERE / "contributions_top5_dynamic.csv", index=False)

    # Charts
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(result["strat_eq"].index, result["strat_eq"].values, label="Top-5 dynamic (20% hysteresis)", linewidth=1.5)
    ax.plot(result["spy_eq"].index,   result["spy_eq"].values,   label="SPY buy & hold",                 linewidth=1.5, alpha=0.8)
    ax.set_title(f"Equity Curve: Dynamic Top-{TOP_N} (20% hysteresis) vs SPY")
    ax.set_xlabel("Date"); ax.set_ylabel("Portfolio Value (USD)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / "equity_curve_top5_dynamic.png", dpi=120); plt.close(fig)

    dd = result["strat_eq"] / result["strat_eq"].cummax() - 1
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    ax.set_title("Dynamic Top-5 Strategy Drawdown")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(HERE / "drawdown_top5_dynamic.png", dpi=120); plt.close(fig)

    # Metrics text
    sm = metrics(result["strat_eq"])
    pm = metrics(result["spy_eq"])

    def fmt_pct(x): return "n/a" if pd.isna(x) else f"{x*100:>9.2f}%"
    def fmt_num(x): return "n/a" if pd.isna(x) else f"{x:>10.3f}"

    n_swaps = sum(1 for t in result["trade_log"] if t["action"] == "sell")
    lines = []
    lines.append("=" * 72)
    lines.append(f"  Dynamic Top-{TOP_N} (>={int((SWAP_THRESHOLD-1)*100)}% hysteresis) -- 2015-01-02 -> 2025-12-31")
    lines.append("=" * 72)
    lines.append(f"  Candidate universe: {', '.join(CANDIDATES)}")
    lines.append(f"  Swap rule: replace lowest holding with non-holding if mcap > {SWAP_THRESHOLD}x holding")
    lines.append("=" * 72)
    lines.append(f"{'Metric':<28}{'Strategy':>20}{'SPY':>20}")
    lines.append("-" * 72)
    lines.append(f"{'Total Return':<28}{fmt_pct(sm['total_return']):>20}{fmt_pct(pm['total_return']):>20}")
    lines.append(f"{'CAGR':<28}{fmt_pct(sm['cagr']):>20}{fmt_pct(pm['cagr']):>20}")
    lines.append(f"{'Sharpe (rf=4.5%)':<28}{fmt_num(sm['sharpe']):>20}{fmt_num(pm['sharpe']):>20}")
    lines.append(f"{'Sortino':<28}{fmt_num(sm['sortino']):>20}{fmt_num(pm['sortino']):>20}")
    lines.append(f"{'Max Drawdown':<28}{fmt_pct(sm['max_dd']):>20}{fmt_pct(pm['max_dd']):>20}")
    lines.append("-" * 72)
    lines.append(f"Total swaps                  : {n_swaps}")
    lines.append(f"Trade-log rows               : {len(result['trade_log'])}")
    lines.append(f"Final holdings               : {', '.join(sorted(result['final_holdings'].keys()))}")
    lines.append(f"Final cash residual          : ${result['final_cash']:,.2f}")
    lines.append("")
    lines.append(f"Per-holding values at end ({sm['total_return']*100:.1f}% portfolio return):")
    for c in result["contributions"]:
        lines.append(f"  {c['ticker']:<6}  shares={c['shares']:>7,}  final_px=${c['final_price']:>8,.2f}  end_value=${c['end_value']:>14,.2f}")
    lines.append("=" * 72)

    body = "\n".join(lines)
    (HERE / "metrics_summary_top5_dynamic.txt").write_text(body, encoding="utf-8")
    print("\n" + body + "\n")


if __name__ == "__main__":
    main()
