"""
mcap_leader.py — Always-hold-the-#1-by-market-cap strategy, 2015→2025.

Strategy
--------
On the first trading day of 2015, identify the company in the candidate set
with the highest market cap (raw_close * shares_outstanding) and put 100% of
capital into it at that day's open. After that, at every daily close,
recompute market cap for all candidates. If a different company has
overtaken the current holding, sell everything at the NEXT day's open and
buy the new leader with the proceeds (after commission + slippage).

Same cost model as the partial-trim backtests:
  - 0.1% commission per side
  - 0.05% slippage per side

Candidate universe
------------------
Limited to ~15 US mega-caps that have been plausibly #1 or close to #1 by
market cap during 2015-2025. Including the full S&P 500 would require 500
extra API calls to yfinance for shares-outstanding history, and the #1 spot
has only ever rotated among ~6 companies in this period anyway.

Data caveat
-----------
yfinance's `get_shares_full()` only returns history back to ~2015-10. For
the first ten months of 2015 we back-fill from the earliest available value
for each ticker. Shares outstanding moves slowly (a few percent per year
from buybacks/issuance and big jumps only on splits, which all candidates
had well-tracked), so the back-fill is a small error for ranking purposes
and won't change which company is #1 (AAPL was clearly #1 throughout
early 2015 and the runners-up were far behind).

Note on Alphabet: we use GOOGL (Class A) shares only. Counting both share
classes would roughly double Alphabet's apparent market cap. Alphabet was
never plausibly #1 in this window so it doesn't affect the result.
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
START_DATE  = "2015-01-01"
END_DATE    = "2025-12-31"
ENTRY_DATE  = "2015-01-02"
COMMISSION_PCT = 0.001
SLIPPAGE_PCT   = 0.0005
RISK_FREE_RATE = 0.045
BENCHMARK = "SPY"

CANDIDATES = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",   # genuine #1 contenders
    "META", "TSLA", "BRK-B",                   # near-top a few times
    "XOM",                                     # was top-5 in 2014-2015
    "JPM", "V", "JNJ", "WMT", "UNH", "PG",     # safety net, top-10 throughout
]

HERE = Path(__file__).resolve().parent
PRICE_CACHE  = HERE / "prices_raw_cache.parquet"
SHARES_CACHE = HERE / "shares_cache.parquet"


# --- Data ---
def fetch_prices() -> pd.DataFrame:
    """Raw (unadjusted) prices + adjusted close for P&L tracking."""
    if PRICE_CACHE.exists():
        print(f"Loading price cache {PRICE_CACHE.name}")
        return pd.read_parquet(PRICE_CACHE)
    tickers = CANDIDATES + [BENCHMARK]
    print(f"Downloading raw prices for {len(tickers)} tickers...")
    raw = yf.download(tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=False, group_by="ticker",
                      threads=True, progress=True)
    # flatten to: column = "{ticker}|{field}" for parquet compat
    panels = {}
    for tk in tickers:
        if tk not in raw.columns.get_level_values(0):
            continue
        sub = raw[tk]
        if not {"Open", "Close", "Adj Close"}.issubset(sub.columns):
            continue
        panels[tk] = sub[["Open", "Close", "Adj Close"]]
    combined = pd.concat(panels, axis=1)
    combined.columns = [f"{t}|{f}" for t, f in combined.columns]
    combined.to_parquet(PRICE_CACHE)
    return combined


def fetch_shares() -> pd.DataFrame:
    """Historical shares outstanding for each candidate, daily-aligned."""
    if SHARES_CACHE.exists():
        print(f"Loading shares cache {SHARES_CACHE.name}")
        return pd.read_parquet(SHARES_CACHE)
    print(f"Downloading shares outstanding for {len(CANDIDATES)} tickers...")
    series = {}
    for tk in CANDIDATES:
        try:
            s = yf.Ticker(tk).get_shares_full(start=START_DATE, end=END_DATE)
            if s is None or len(s) == 0:
                print(f"  {tk}: NO DATA")
                continue
            # de-dup any timestamp collisions, keep last
            s = s[~s.index.duplicated(keep="last")]
            series[tk] = s
            print(f"  {tk}: {len(s)} entries, range {s.index.min().date()} -> {s.index.max().date()}")
        except Exception as e:
            print(f"  {tk}: ERROR {e}")
    df = pd.DataFrame(series)
    # Drop the timezone info to allow safe alignment to trading-day index later
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    df.to_parquet(SHARES_CACHE)
    return df


def get_split_factor(tk: str, index: pd.DatetimeIndex) -> pd.Series:
    """For each date D in index, return the cumulative-forward split factor:
    product of all split ratios with split_date > D.

    Use case: yfinance's `Close` is split-adjusted, so older dates report
    a "current-share-equivalent" price. To express *shares outstanding* in
    the same current-share-equivalent units (so price * shares = real
    market cap), multiply the actual-at-time share count by this factor.

    Example: V had a 4-for-1 split on 2015-03-19. For dates before that,
    factor = 4.0 — so Jan 2015's actual 608M shares becomes 2.43B in
    current-share-equivalent units (matching the price scale).
    """
    splits = yf.Ticker(tk).splits
    if splits is None or len(splits) == 0:
        return pd.Series(1.0, index=index)
    splits.index = pd.to_datetime(splits.index).tz_localize(None)
    factor = pd.Series(1.0, index=index)
    for split_date, ratio in splits.items():
        # Include the split day itself — yfinance reports shares on a split
        # day in PRE-split units, so the value still needs the split factor.
        mask = factor.index <= split_date
        factor.loc[mask] *= float(ratio)
    return factor


def daily_market_cap(prices: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Market cap = split-adjusted close × split-adjusted shares.

    Both sides must be in the same units (here: current-share-equivalent).
    yfinance gives split-adjusted prices for free; we convert its actual-
    at-time shares into split-adjusted shares by multiplying by the
    forward-cumulative split factor at each data point's date.

    The result is invariant across splits — for V: $66 split-adj price ×
    2.43B split-adj shares = $161B, regardless of whether we look before
    or after the March 2015 split.
    """
    raw_close = pd.DataFrame({tk: prices[f"{tk}|Close"] for tk in CANDIDATES})
    print("Computing split-adjusted shares for each candidate...")
    mcap = pd.DataFrame(index=raw_close.index, columns=CANDIDATES, dtype="float64")
    for tk in CANDIDATES:
        if tk not in shares.columns:
            continue
        shares_ts = shares[tk].dropna()
        if len(shares_ts) == 0:
            continue
        # Per-data-point split-adj factor (product of splits ON OR AFTER that date)
        factor_at_shares = get_split_factor(tk, shares_ts.index)
        split_adj = shares_ts * factor_at_shares.reindex(shares_ts.index)
        # yfinance occasionally returns wildly off values on a split day
        # (e.g., TSLA 2020-08-31 reported 4.66B, which is neither pre- nor
        # post-split). Drop any point whose split-adj value is >2x or <0.5x
        # the median of the rest. Shares-outstanding changes slowly so any
        # such jump is a data error, not a real corporate event.
        med = split_adj.median()
        if med > 0:
            ratio = split_adj / med
            outlier_mask = (ratio > 2.0) | (ratio < 0.5)
            n_drop = int(outlier_mask.sum())
            if n_drop > 0:
                print(f"  {tk}: dropped {n_drop} outlier data point(s) "
                      f"(values {split_adj[outlier_mask].round(0).tolist()})")
                split_adj = split_adj[~outlier_mask]
        sa_daily = split_adj.reindex(raw_close.index).ffill().bfill()
        mcap[tk] = raw_close[tk] * sa_daily
    return mcap


# --- Strategy simulation ---
def simulate(prices: pd.DataFrame, mcap: pd.DataFrame) -> dict:
    """Walk forward day-by-day, switching to the new leader at next-day open."""
    dates = mcap.index
    entry_idx = int(dates.searchsorted(pd.Timestamp(ENTRY_DATE)))
    n_days = len(dates)

    # Pre-extract series as numpy for speed
    raw_close = {tk: prices[f"{tk}|Close"].values for tk in CANDIDATES}
    raw_open  = {tk: prices[f"{tk}|Open"].values  for tk in CANDIDATES}
    adj_close = {tk: prices[f"{tk}|Adj Close"].values for tk in CANDIDATES}
    # adjusted open = raw_open * (adj_close / raw_close) — preserves split/div adjustment
    adj_open  = {tk: (prices[f"{tk}|Open"] * prices[f"{tk}|Adj Close"] / prices[f"{tk}|Close"]).values
                 for tk in CANDIDATES}

    # Determine leader on entry day (using entry-day market cap)
    initial_leader = str(mcap.iloc[entry_idx].idxmax())

    # Buy initial leader at entry-day adjusted open
    exec_price = adj_open[initial_leader][entry_idx] * (1 + SLIPPAGE_PCT)
    per_share_cost = exec_price * (1 + COMMISSION_PCT)
    holder_shares = int(math.floor(STARTING_CAPITAL / per_share_cost))
    cost = holder_shares * per_share_cost
    cash = STARTING_CAPITAL - cost
    current_holder = initial_leader

    trades = [dict(
        date=dates[entry_idx].strftime("%Y-%m-%d"),
        ticker=initial_leader, action="buy",
        shares=holder_shares, exec_price=round(exec_price, 4),
        gross=round(holder_shares * exec_price, 2),
        commission=round(holder_shares * exec_price * COMMISSION_PCT, 2),
        cash_after=round(cash, 2),
        reason="initial_entry",
        leader_mcap_billions=round(float(mcap.iloc[entry_idx][initial_leader]) / 1e9, 1),
    )]

    # SPY benchmark setup
    spy_adj_close = prices[f"{BENCHMARK}|Adj Close"].values
    spy_raw_open  = prices[f"{BENCHMARK}|Open"].values
    spy_raw_close = prices[f"{BENCHMARK}|Close"].values
    spy_adj_open  = spy_raw_open * (spy_adj_close / spy_raw_close)
    spy_exec = spy_adj_open[entry_idx] * (1 + SLIPPAGE_PCT)
    spy_per_share = spy_exec * (1 + COMMISSION_PCT)
    spy_shares = int(math.floor(STARTING_CAPITAL / spy_per_share))
    spy_cash = STARTING_CAPITAL - spy_shares * spy_per_share

    strat_eq = np.full(n_days, STARTING_CAPITAL, dtype=np.float64)
    spy_eq   = np.full(n_days, STARTING_CAPITAL, dtype=np.float64)
    holder_history = []  # (date, ticker, mcap_billions)

    for i in range(n_days):
        if i < entry_idx:
            continue
        # Mark to market at today's adjusted close
        h_close = adj_close[current_holder][i]
        if not np.isnan(h_close):
            strat_eq[i] = holder_shares * h_close + cash
        else:
            strat_eq[i] = strat_eq[i - 1]
        sc = spy_adj_close[i]
        if not np.isnan(sc):
            spy_eq[i] = spy_shares * sc + spy_cash

        # Check leader at today's close
        row = mcap.iloc[i]
        today_leader = str(row.idxmax())
        holder_history.append((dates[i], current_holder, float(row[current_holder]) / 1e9))

        # Switch at next day's open if leader changed
        if today_leader != current_holder and i + 1 < n_days:
            ni = i + 1
            sell_price = adj_open[current_holder][ni] * (1 - SLIPPAGE_PCT)
            if np.isnan(sell_price):
                continue
            sell_gross = holder_shares * sell_price
            sell_comm = sell_gross * COMMISSION_PCT
            sell_proceeds = sell_gross - sell_comm
            cash += sell_proceeds
            trades.append(dict(
                date=dates[ni].strftime("%Y-%m-%d"),
                ticker=current_holder, action="sell",
                shares=holder_shares, exec_price=round(sell_price, 4),
                gross=round(sell_gross, 2), commission=round(sell_comm, 2),
                cash_after=round(cash, 2),
                reason=f"surpassed_by_{today_leader}",
                leader_mcap_billions=round(float(row[today_leader]) / 1e9, 1),
            ))

            buy_price = adj_open[today_leader][ni] * (1 + SLIPPAGE_PCT)
            if np.isnan(buy_price):
                continue
            per_share_cost = buy_price * (1 + COMMISSION_PCT)
            new_shares = int(math.floor(cash / per_share_cost))
            buy_cost = new_shares * per_share_cost
            cash -= buy_cost
            trades.append(dict(
                date=dates[ni].strftime("%Y-%m-%d"),
                ticker=today_leader, action="buy",
                shares=new_shares, exec_price=round(buy_price, 4),
                gross=round(new_shares * buy_price, 2),
                commission=round(new_shares * buy_price * COMMISSION_PCT, 2),
                cash_after=round(cash, 2),
                reason=f"new_leader",
                leader_mcap_billions=round(float(row[today_leader]) / 1e9, 1),
            ))
            holder_shares = new_shares
            current_holder = today_leader

    return dict(
        trades=trades,
        strat_eq=pd.Series(strat_eq, index=dates, name="Strategy"),
        spy_eq=pd.Series(spy_eq, index=dates, name="SPY"),
        holder_history=holder_history,
        final_holder=current_holder,
        final_shares=holder_shares,
        final_cash=cash,
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


# --- Output ---
def write_outputs(result: dict) -> None:
    trades_df = pd.DataFrame(result["trades"])
    trades_df.to_csv(HERE / "trade_log_mcap.csv", index=False)
    print(f"Wrote trade_log_mcap.csv ({len(trades_df)} rows)")

    eq_df = pd.DataFrame({"strategy": result["strat_eq"], "spy": result["spy_eq"]})
    eq_df.index.name = "date"
    eq_df.to_csv(HERE / "equity_curve_mcap.csv")
    print(f"Wrote equity_curve_mcap.csv")

    # Holder time line — summarize who was held when
    hist = result["holder_history"]
    if hist:
        segments = []
        prev = hist[0][1]
        seg_start = hist[0][0]
        for d, tk, mc in hist[1:]:
            if tk != prev:
                segments.append((seg_start, d, prev))
                seg_start = d
                prev = tk
        segments.append((seg_start, hist[-1][0], prev))
        seg_df = pd.DataFrame(segments, columns=["from_date", "to_date", "ticker"])
        seg_df["days_held"] = (seg_df["to_date"] - seg_df["from_date"]).dt.days
        seg_df.to_csv(HERE / "holder_timeline.csv", index=False)
        print(f"Wrote holder_timeline.csv ({len(seg_df)} segments)")

    # Charts
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(result["strat_eq"].index, result["strat_eq"].values, label="MCap-leader strategy", linewidth=1.5)
    ax.plot(result["spy_eq"].index,   result["spy_eq"].values,   label="SPY buy & hold",       linewidth=1.5, alpha=0.8)
    ax.set_title("Equity Curve: Always-Hold-#1-By-Market-Cap vs SPY")
    ax.set_xlabel("Date"); ax.set_ylabel("Portfolio Value (USD)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / "equity_curve_mcap.png", dpi=120); plt.close(fig)
    print("Wrote equity_curve_mcap.png")

    dd = result["strat_eq"] / result["strat_eq"].cummax() - 1
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    ax.set_title("MCap-leader strategy drawdown")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(HERE / "drawdown_mcap.png", dpi=120); plt.close(fig)
    print("Wrote drawdown_mcap.png")

    # Metrics text
    sm = metrics(result["strat_eq"], "Strategy")
    pm = metrics(result["spy_eq"], "SPY")

    def fmt_pct(x): return "n/a" if pd.isna(x) else f"{x*100:>9.2f}%"
    def fmt_num(x): return "n/a" if pd.isna(x) else f"{x:>10.3f}"

    lines = []
    lines.append("=" * 72)
    lines.append("  Always-Hold-#1-By-Market-Cap Strategy — 2015-01-02 -> 2025-12-31")
    lines.append("=" * 72)
    lines.append(f"  Candidate universe ({len(CANDIDATES)} tickers): {', '.join(CANDIDATES)}")
    lines.append(f"  Starting capital: ${STARTING_CAPITAL:,}")
    lines.append("=" * 72)
    lines.append(f"{'Metric':<28}{'Strategy':>20}{'SPY':>20}")
    lines.append("-" * 72)
    lines.append(f"{'Total Return':<28}{fmt_pct(sm['total_return']):>20}{fmt_pct(pm['total_return']):>20}")
    lines.append(f"{'CAGR':<28}{fmt_pct(sm['cagr']):>20}{fmt_pct(pm['cagr']):>20}")
    lines.append(f"{'Sharpe (rf=4.5%)':<28}{fmt_num(sm['sharpe']):>20}{fmt_num(pm['sharpe']):>20}")
    lines.append(f"{'Sortino':<28}{fmt_num(sm['sortino']):>20}{fmt_num(pm['sortino']):>20}")
    lines.append(f"{'Max Drawdown':<28}{fmt_pct(sm['max_dd']):>20}{fmt_pct(pm['max_dd']):>20}")
    lines.append("-" * 72)
    lines.append(f"Total switches               : {int((trades_df['action']=='buy').sum()) - 1}")
    lines.append(f"Trade-log rows (incl. entry) : {len(trades_df)}")
    lines.append(f"Final holding                : {result['final_holder']}")
    lines.append(f"  Final shares               : {result['final_shares']:,}")
    lines.append(f"  Final cash residual        : ${result['final_cash']:,.2f}")
    lines.append("=" * 72)

    body = "\n".join(lines)
    (HERE / "metrics_summary_mcap.txt").write_text(body, encoding="utf-8")
    print("\n" + body + "\n")
    print("Wrote metrics_summary_mcap.txt")


def main():
    prices = fetch_prices()
    shares = fetch_shares()
    mcap = daily_market_cap(prices, shares)
    print(f"\nLeader on entry day {ENTRY_DATE}: {mcap.iloc[mcap.index.searchsorted(pd.Timestamp(ENTRY_DATE))].idxmax()}")
    print(f"Leader on final day {mcap.index[-1].date()}: {mcap.iloc[-1].idxmax()}")
    result = simulate(prices, mcap)
    write_outputs(result)


if __name__ == "__main__":
    main()
