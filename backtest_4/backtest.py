"""
backtest.py — S&P 500 momentum partial-trim backtest, 2015-01-02 to 2024-12-31.

Strategy
--------
Equal-weight buy every current S&P 500 stock at the open on 2015-01-02. For
each stock, track a per-name reference price (initially that day's close).
At every daily close:

  - If close has risen >= +10% vs reference -> queue a profit-take sale.
  - If close has fallen <= -5%  vs reference -> queue a stop-loss sale.

Sales execute at the NEXT day's open. Trim size is a PERCENTAGE of the
position's current value (not a fixed dollar amount): 1% on profit-takes
and 0.5% on stop-losses, with a floor of 1 share so small positions can
still trim. This scales naturally as positions grow — a 10x'd NVDA still
trims a meaningful number of shares per trigger. After every actual partial
sale the reference price resets to the trigger-day close. When a remaining
position falls below $1 it is fully closed and never re-entered.

Costs
-----
Commission: 0.1% of trade value per side. Slippage: 0.05% (buys fill higher,
sells fill lower). No short selling, no leverage. Cash from sales sits idle.

Known limitations
-----------------
- Survivorship bias: we use the *current* S&P 500 list, so names that were
  removed/delisted between 2015 and 2024 are excluded entirely. Real-world
  results would be lower.
- Whole-share rounding: target dollar sells are floored to integer shares,
  so a few high-priced names may sell less (or zero) per trigger.
- Adjusted prices: yfinance auto_adjust=True folds splits and dividends into
  prices, so dividends never appear as cash. Total return slightly understates
  a real broker's experience.
- Delisting proxy: a ticker whose price series ends before 2024-12-31 is
  force-closed at its last available close.
- No re-entry: capital from a fully-closed position sits in cash earning 0%.

Usage
-----
    python backtest.py             # uses cached prices if present
    python backtest.py --refresh   # forces a fresh yfinance download
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
STARTING_CAPITAL = 100_000

START_DATE = "2015-01-01"
END_DATE   = "2025-12-31"
ENTRY_DATE = "2015-01-02"

PROFIT_TAKE_PCT  = 0.10
STOP_LOSS_PCT    = -0.05
# Trim sizes are a percentage of the position's CURRENT value (not a fixed
# dollar amount tied to starting capital). This scales naturally as positions
# grow — so a NVDA position that's 10x'd still trims a meaningful number of
# shares per trigger instead of rounding to zero. Ratio preserved at 2:1
# (profit-take is twice as aggressive as stop-loss) to match the spirit of
# the original $10 / $5 sizing.
PROFIT_SELL_PCT  = 0.001     # sell 0.1%  of position value per profit-take
STOP_SELL_PCT    = 0.0005    # sell 0.05% of position value per stop-loss
MIN_POSITION_USD = 1.00      # full-close threshold

COMMISSION_PCT = 0.001       # 0.1% per side
SLIPPAGE_PCT   = 0.0005      # 0.05% (buys higher, sells lower)
RISK_FREE_RATE = 0.045       # annualized, used for Sharpe / Sortino

BENCHMARK_TICKER = "SPY"

OUTPUT_DIR   = Path(__file__).resolve().parent
CACHE_PATH   = OUTPUT_DIR / "prices_cache.parquet"
TRADE_LOG    = OUTPUT_DIR / "trade_log.csv"
METRICS_TXT  = OUTPUT_DIR / "metrics_summary.txt"
METRICS_CSV  = OUTPUT_DIR / "metrics_summary.csv"
EQUITY_CSV   = OUTPUT_DIR / "equity_curve.csv"
EQUITY_PNG   = OUTPUT_DIR / "equity_curve.png"
DRAWDOWN_PNG = OUTPUT_DIR / "drawdown.png"

# Liquid mega-caps for --smoke mode (no full 500-ticker download required).
SMOKE_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                 "META", "TSLA", "JPM", "JNJ", "V"]


# ============================================================================
# 2. DATA
# ============================================================================
def get_sp500_tickers() -> list[str]:
    # Survivorship bias: this is the *current* S&P 500, not the 2015 list.
    # Wikipedia blocks requests without a browser User-Agent, so we fetch
    # the HTML ourselves and hand it to pandas (wrapped in StringIO for
    # compatibility with pandas >= 2.1).
    import urllib.request
    from io import StringIO
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8")
    table = pd.read_html(StringIO(html))[0]
    syms = table["Symbol"].astype(str).str.strip().tolist()
    return [s.replace(".", "-") for s in syms]


def load_prices(tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame with MultiIndex columns (ticker, field) where field
    is Open or Close. Cached to parquet to avoid re-downloading."""
    all_tix = sorted(set(tickers + [BENCHMARK_TICKER]))

    if CACHE_PATH.exists() and not refresh:
        cached = pd.read_parquet(CACHE_PATH)
        cached.columns = pd.MultiIndex.from_tuples(
            [tuple(c.split("|")) for c in cached.columns], names=["Ticker", "Field"]
        )
        cached = cached.sort_index().sort_index(axis=1)

        # Reject the cache if it doesn't cover the requested window. Without
        # this check, bumping END_DATE in CONFIGURATION would silently produce
        # a backtest that ends wherever the old cache ends.
        requested_end = pd.Timestamp(END_DATE)
        cache_end = cached.index.max()
        if cache_end < requested_end - pd.Timedelta(days=7):
            print(f"Cache ends {cache_end.date()} but END_DATE is {END_DATE} — "
                  f"forcing refresh to cover the requested window.")
        else:
            print(f"Loading cached prices from {CACHE_PATH.name} "
                  f"({cached.index.min().date()} -> {cache_end.date()})")
            return cached

    print(f"Downloading {len(all_tix)} tickers from yfinance ({START_DATE} -> {END_DATE})...")
    raw = yf.download(
        all_tix,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=True,
    )

    panels = {}
    for tk in all_tix:
        if tk not in raw.columns.get_level_values(0):
            continue
        sub = raw[tk]
        if "Open" not in sub.columns or "Close" not in sub.columns:
            continue
        panels[tk] = sub[["Open", "Close"]]

    combined = pd.concat(panels, axis=1)
    combined.columns.names = ["Ticker", "Field"]
    combined = combined.sort_index().sort_index(axis=1)

    # parquet column names must be strings, so flatten the MultiIndex for storage.
    to_save = combined.copy()
    to_save.columns = [f"{t}|{f}" for t, f in combined.columns]
    to_save.to_parquet(CACHE_PATH)
    print(f"Cached to {CACHE_PATH.name}")
    return combined


# ============================================================================
# 3. PORTFOLIO INIT
# ============================================================================
def initialize(opens: pd.DataFrame, closes: pd.DataFrame):
    """Equal-weight buy every eligible ticker at first available open on/after
    ENTRY_DATE. Returns numpy-backed state plus initial cash and a list of
    entry trade-log rows."""
    entry_anchor = opens.index.searchsorted(pd.Timestamp(ENTRY_DATE))
    if entry_anchor >= len(opens):
        raise RuntimeError(f"No trading day on/after {ENTRY_DATE}")

    tickers, entry_indices = [], []
    for tk in opens.columns:
        if tk == BENCHMARK_TICKER:
            continue
        sub_o = opens[tk].iloc[entry_anchor:entry_anchor + 5]
        sub_c = closes[tk].iloc[entry_anchor:entry_anchor + 5]
        ok = (sub_o.notna() & sub_c.notna()).values
        if ok.any():
            tickers.append(tk)
            entry_indices.append(entry_anchor + int(ok.argmax()))

    n = len(tickers)
    if n == 0:
        raise RuntimeError("No eligible tickers on entry date")
    alloc = STARTING_CAPITAL / n
    print(f"{n} eligible tickers / {len(opens.columns) - 1} S&P 500 names")
    print(f"Per-stock allocation: ${alloc:,.2f}")

    opens_np  = opens[tickers].values
    closes_np = closes[tickers].values

    shares      = np.zeros(n, dtype=np.int64)
    reference   = np.full(n, np.nan)
    cost_basis  = np.zeros(n)
    proceeds    = np.zeros(n)
    closed      = np.zeros(n, dtype=bool)
    entry_idx   = np.array(entry_indices, dtype=np.int64)

    # Last valid (non-NaN) close index per ticker — used as delisting proxy.
    last_valid = np.full(n, len(opens) - 1, dtype=np.int64)
    for j, tk in enumerate(tickers):
        lvi = closes[tk].last_valid_index()
        if lvi is not None:
            last_valid[j] = closes.index.get_loc(lvi)

    entry_log: list[dict] = []
    total_cost = 0.0
    for j in range(n):
        i = entry_idx[j]
        raw_open = opens_np[i, j]
        exec_price = raw_open * (1 + SLIPPAGE_PCT)
        sh = math.floor(alloc / exec_price)
        if sh <= 0:
            closed[j] = True
            continue
        gross = sh * exec_price
        commission = gross * COMMISSION_PCT
        cost = gross + commission
        total_cost += cost
        shares[j] = sh
        reference[j] = closes_np[i, j]   # reference starts at entry-day close
        cost_basis[j] = cost
        entry_log.append(dict(
            date=opens.index[i].strftime("%Y-%m-%d"),
            ticker=tickers[j],
            trigger_type="entry",
            shares=sh,
            exec_price=round(exec_price, 4),
            dollar_amount=round(gross, 2),
            commission=round(commission, 2),
            reference_at_trigger=np.nan,
            cash_after=np.nan,
            portfolio_value=np.nan,
        ))

    cash = STARTING_CAPITAL - total_cost
    print(f"Initial cash after entries: ${cash:,.2f}")

    state = dict(
        tickers=tickers,
        opens=opens_np,
        closes=closes_np,
        shares=shares,
        reference=reference,
        cost_basis=cost_basis,
        proceeds=proceeds,
        closed=closed,
        entry_idx=entry_idx,
        last_valid=last_valid,
    )
    return state, cash, entry_log


# ============================================================================
# 4. BACKTEST LOOP
# ============================================================================
def run_backtest(prices: pd.DataFrame):
    opens  = prices.xs("Open",  level="Field", axis=1)
    closes = prices.xs("Close", level="Field", axis=1)

    state, cash, trade_log = initialize(opens, closes)
    tickers     = state["tickers"]
    opens_np    = state["opens"]
    closes_np   = state["closes"]
    shares      = state["shares"]
    reference   = state["reference"]
    proceeds    = state["proceeds"]
    closed      = state["closed"]
    entry_idx   = state["entry_idx"]
    last_valid  = state["last_valid"]
    n_tix       = len(tickers)
    n_days      = len(opens)

    dates = opens.index
    spy_open_series  = opens[BENCHMARK_TICKER].values
    spy_close_series = closes[BENCHMARK_TICKER].values

    # SPY benchmark: buy at first valid open on/after ENTRY_DATE.
    spy_anchor = int(opens.index.searchsorted(pd.Timestamp(ENTRY_DATE)))
    while spy_anchor < n_days and (
        np.isnan(spy_open_series[spy_anchor]) or np.isnan(spy_close_series[spy_anchor])
    ):
        spy_anchor += 1
    spy_exec   = spy_open_series[spy_anchor] * (1 + SLIPPAGE_PCT)
    spy_shares = math.floor(STARTING_CAPITAL / spy_exec)
    spy_cost   = spy_shares * spy_exec * (1 + COMMISSION_PCT)
    spy_cash   = STARTING_CAPITAL - spy_cost

    strat_equity = np.full(n_days, STARTING_CAPITAL, dtype=np.float64)
    spy_equity   = np.full(n_days, STARTING_CAPITAL, dtype=np.float64)
    last_spy_close = spy_close_series[spy_anchor]

    earliest_entry = int(entry_idx.min())

    # Pending fills queued at close of day t, executed at open of day t+1.
    # ticker_col -> (trigger_type, reference_at_trigger_close)
    pending: dict[int, tuple[str, float]] = {}

    for i in range(n_days):
        # ---- benchmark mark-to-market ----
        if i >= spy_anchor:
            c = spy_close_series[i]
            if not np.isnan(c):
                last_spy_close = c
            spy_equity[i] = spy_shares * last_spy_close + spy_cash

        if i < earliest_entry:
            continue

        # ---- 4a. execute pending fills at today's open ----
        if pending:
            for j in list(pending.keys()):
                if closed[j]:
                    pending.pop(j, None)
                    continue
                trig_type, ref_at_trig = pending[j]
                today_open = opens_np[i, j]
                if np.isnan(today_open):
                    # no fill today -> keep pending, try again tomorrow
                    continue
                exec_price = today_open * (1 - SLIPPAGE_PCT)

                # Percentage of CURRENT position value. shares*pct collapses to a
                # share-count directly (target_usd / exec_price == shares * pct
                # since target_usd = shares*exec_price*pct). The max(1, ...) floor
                # ensures even small positions can trim — without it a position
                # of, say, 50 shares at PT=1% would round to 0 shares forever.
                pct = PROFIT_SELL_PCT if trig_type == "profit_take" else STOP_SELL_PCT
                target_shares = int(math.floor(shares[j] * pct))
                sh_sell = min(int(shares[j]), max(1, target_shares))

                gross = sh_sell * exec_price
                commission = gross * COMMISSION_PCT
                got = gross - commission
                shares[j] -= sh_sell
                proceeds[j] += got
                cash += got
                reference[j] = ref_at_trig   # reset to TRIGGER-day close

                trade_log.append(dict(
                    date=dates[i].strftime("%Y-%m-%d"),
                    ticker=tickers[j],
                    trigger_type=trig_type,
                    shares=sh_sell,
                    exec_price=round(exec_price, 4),
                    dollar_amount=round(gross, 2),
                    commission=round(commission, 2),
                    reference_at_trigger=round(ref_at_trig, 4),
                    cash_after=round(cash, 2),
                    portfolio_value=np.nan,
                ))

                # Full close if leftover value below threshold.
                if shares[j] == 0:
                    closed[j] = True
                else:
                    today_close = closes_np[i, j]
                    if not np.isnan(today_close) and shares[j] * today_close < MIN_POSITION_USD:
                        rem = int(shares[j])
                        g2 = rem * exec_price
                        c2 = g2 * COMMISSION_PCT
                        p2 = g2 - c2
                        shares[j] = 0
                        proceeds[j] += p2
                        cash += p2
                        closed[j] = True
                        trade_log.append(dict(
                            date=dates[i].strftime("%Y-%m-%d"),
                            ticker=tickers[j],
                            trigger_type="full_close",
                            shares=rem,
                            exec_price=round(exec_price, 4),
                            dollar_amount=round(g2, 2),
                            commission=round(c2, 2),
                            reference_at_trigger=round(ref_at_trig, 4),
                            cash_after=round(cash, 2),
                            portfolio_value=np.nan,
                        ))

                pending.pop(j, None)

        # ---- 4b. delisting proxy: force-close any ticker past its last valid day ----
        delist_j = np.where(~closed & (i > last_valid) & (shares > 0))[0]
        for j in delist_j:
            lvi = int(last_valid[j])
            last_close = closes_np[lvi, j]
            if np.isnan(last_close):
                closed[j] = True
                continue
            exec_price = last_close * (1 - SLIPPAGE_PCT)
            rem = int(shares[j])
            gross = rem * exec_price
            commission = gross * COMMISSION_PCT
            got = gross - commission
            shares[j] = 0
            proceeds[j] += got
            cash += got
            closed[j] = True
            trade_log.append(dict(
                date=dates[lvi].strftime("%Y-%m-%d"),
                ticker=tickers[j],
                trigger_type="full_close",
                shares=rem,
                exec_price=round(exec_price, 4),
                dollar_amount=round(gross, 2),
                commission=round(commission, 2),
                reference_at_trigger=round(float(reference[j]) if not np.isnan(reference[j]) else 0.0, 4),
                cash_after=round(cash, 2),
                portfolio_value=np.nan,
            ))

        # ---- 4c. mark-to-market + trigger detection at today's close ----
        close_row = closes_np[i, :]
        active = ~closed & (entry_idx <= i) & ~np.isnan(close_row)

        held_value = np.where(active, shares * close_row, 0.0).sum()
        strat_equity[i] = cash + held_value

        # Vectorized trigger evaluation.
        with np.errstate(invalid="ignore", divide="ignore"):
            ret = (close_row - reference) / reference
        eligible_for_trigger = active & ~np.isin(np.arange(n_tix), list(pending.keys()))
        profit_hits = np.where(eligible_for_trigger & (ret >= PROFIT_TAKE_PCT))[0]
        stop_hits   = np.where(eligible_for_trigger & (ret <= STOP_LOSS_PCT)
                               & ~(ret >= PROFIT_TAKE_PCT))[0]

        for j in profit_hits:
            pending[int(j)] = ("profit_take", float(close_row[j]))
        for j in stop_hits:
            pending[int(j)] = ("stop_loss", float(close_row[j]))

    # End-of-run: backfill portfolio_value column in trade_log by date.
    eq_by_date = pd.Series(strat_equity, index=dates)
    for row in trade_log:
        d = pd.Timestamp(row["date"])
        if d in eq_by_date.index:
            row["portfolio_value"] = round(float(eq_by_date.loc[d]), 2)

    strat_eq_series = pd.Series(strat_equity, index=dates, name="Strategy")
    spy_eq_series   = pd.Series(spy_equity,   index=dates, name="SPY")

    summary_state = dict(
        tickers=tickers,
        shares=shares.copy(),
        cost_basis=state["cost_basis"].copy(),
        proceeds=proceeds.copy(),
        closed=closed.copy(),
    )
    return trade_log, strat_eq_series, spy_eq_series, summary_state


# ============================================================================
# 5. METRICS
# ============================================================================
def compute_metrics(equity: pd.Series, name: str) -> dict:
    eq = equity.dropna()
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
        name=name,
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
    )


def trade_stats(trade_log: list, state: dict) -> dict:
    df = pd.DataFrame(trade_log)
    n_trades = len(df)

    # Win rate over fully-closed positions: proceeds > cost basis.
    closed_mask = state["closed"]
    if closed_mask.any():
        cb = state["cost_basis"][closed_mask]
        pr = state["proceeds"][closed_mask]
        win_rate = float((pr > cb).mean())
        n_closed = int(closed_mask.sum())
    else:
        win_rate = float("nan")
        n_closed = 0

    pt = df[df["trigger_type"] == "profit_take"]
    sl = df[df["trigger_type"] == "stop_loss"]

    def avg_move(sub):
        if len(sub) == 0:
            return float("nan")
        return float(((sub["exec_price"] - sub["reference_at_trigger"]) / sub["reference_at_trigger"]).mean())

    return dict(
        total_trades=n_trades,
        win_rate=win_rate,
        avg_gain_profit_take=avg_move(pt),
        avg_loss_stop_loss=avg_move(sl),
        n_closed=n_closed,
        n_profit_takes=len(pt),
        n_stop_losses=len(sl),
    )


# ============================================================================
# 6. OUTPUT
# ============================================================================
def write_outputs(trade_log, strat_eq, spy_eq, state):
    pd.DataFrame(trade_log).to_csv(TRADE_LOG, index=False)
    print(f"Wrote {TRADE_LOG.name} ({len(trade_log):,} rows)")

    # Persist the daily equity curve so validate.py can slice it later
    # without re-running the simulation.
    eq_df = pd.DataFrame({"strategy": strat_eq, "spy": spy_eq})
    eq_df.index.name = "date"
    eq_df.to_csv(EQUITY_CSV)
    print(f"Wrote {EQUITY_CSV.name} ({len(eq_df):,} rows)")

    # equity curve
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(strat_eq.index, strat_eq.values, label="Strategy", linewidth=1.5)
    ax.plot(spy_eq.index,   spy_eq.values,   label="SPY buy & hold", linewidth=1.5, alpha=0.8)
    ax.set_title("Equity Curve: Strategy vs SPY")
    ax.set_xlabel("Date"); ax.set_ylabel("Portfolio Value (USD)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(EQUITY_PNG, dpi=120); plt.close(fig)
    print(f"Wrote {EQUITY_PNG.name}")

    # drawdown
    dd = strat_eq / strat_eq.cummax() - 1
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    ax.plot(dd.index, dd.values, color="darkred", linewidth=0.8)
    ax.set_title("Strategy Drawdown")
    ax.set_xlabel("Date"); ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(DRAWDOWN_PNG, dpi=120); plt.close(fig)
    print(f"Wrote {DRAWDOWN_PNG.name}")

    strat_m = compute_metrics(strat_eq, "Strategy")
    spy_m   = compute_metrics(spy_eq,   "SPY")
    ts      = trade_stats(trade_log, state)

    def fmt_pct(x): return "n/a" if pd.isna(x) else f"{x*100:>9.2f}%"
    def fmt_num(x): return "n/a" if pd.isna(x) else f"{x:>10.3f}"

    lines = []
    lines.append("=" * 72)
    lines.append("  S&P 500 Momentum Partial-Trim Backtest — Summary")
    lines.append(f"  {START_DATE}  ->  {END_DATE}    Starting capital: ${STARTING_CAPITAL:,}")
    lines.append("=" * 72)
    lines.append(f"{'Metric':<28}{'Strategy':>20}{'SPY':>20}")
    lines.append("-" * 72)
    lines.append(f"{'Total Return':<28}{fmt_pct(strat_m['total_return']):>20}{fmt_pct(spy_m['total_return']):>20}")
    lines.append(f"{'CAGR':<28}{fmt_pct(strat_m['cagr']):>20}{fmt_pct(spy_m['cagr']):>20}")
    lines.append(f"{'Sharpe (rf=4.5%)':<28}{fmt_num(strat_m['sharpe']):>20}{fmt_num(spy_m['sharpe']):>20}")
    lines.append(f"{'Sortino':<28}{fmt_num(strat_m['sortino']):>20}{fmt_num(spy_m['sortino']):>20}")
    lines.append(f"{'Max Drawdown':<28}{fmt_pct(strat_m['max_drawdown']):>20}{fmt_pct(spy_m['max_drawdown']):>20}")
    lines.append("-" * 72)
    lines.append(f"Total trades executed       : {ts['total_trades']:,}")
    lines.append(f"   profit-take fills        : {ts['n_profit_takes']:,}")
    lines.append(f"   stop-loss fills          : {ts['n_stop_losses']:,}")
    lines.append(f"   fully closed positions   : {ts['n_closed']:,}")
    lines.append(f"Win rate (closed positions) : {fmt_pct(ts['win_rate'])}")
    lines.append(f"Avg gain on profit-take fill: {fmt_pct(ts['avg_gain_profit_take'])}")
    lines.append(f"Avg loss on stop-loss fill  : {fmt_pct(ts['avg_loss_stop_loss'])}")
    lines.append("=" * 72)

    body = "\n".join(lines)
    METRICS_TXT.write_text(body)
    print("\n" + body + "\n")
    print(f"Wrote {METRICS_TXT.name}")

    # Machine-readable single-row metrics file — used by validate.py to
    # build a side-by-side fold comparison.
    row = dict(
        period=f"{START_DATE}__{END_DATE}",
        strat_total_return=strat_m["total_return"],
        strat_cagr=strat_m["cagr"],
        strat_sharpe=strat_m["sharpe"],
        strat_sortino=strat_m["sortino"],
        strat_max_dd=strat_m["max_drawdown"],
        spy_total_return=spy_m["total_return"],
        spy_cagr=spy_m["cagr"],
        spy_sharpe=spy_m["sharpe"],
        spy_sortino=spy_m["sortino"],
        spy_max_dd=spy_m["max_drawdown"],
        total_trades=ts["total_trades"],
        profit_take_fills=ts["n_profit_takes"],
        stop_loss_fills=ts["n_stop_losses"],
        closed_positions=ts["n_closed"],
        win_rate=ts["win_rate"],
        avg_gain_profit_take=ts["avg_gain_profit_take"],
        avg_loss_stop_loss=ts["avg_loss_stop_loss"],
    )
    pd.DataFrame([row]).to_csv(METRICS_CSV, index=False)
    print(f"Wrote {METRICS_CSV.name}")


# ============================================================================
# ENTRY POINT
# ============================================================================
def main():
    global START_DATE, END_DATE, ENTRY_DATE
    global CACHE_PATH, TRADE_LOG, METRICS_TXT, METRICS_CSV, EQUITY_CSV, EQUITY_PNG, DRAWDOWN_PNG

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start",   default=START_DATE, help="Backtest start date YYYY-MM-DD")
    parser.add_argument("--end",     default=END_DATE,   help="Backtest end date YYYY-MM-DD")
    parser.add_argument("--entry",   default=ENTRY_DATE, help="Entry date YYYY-MM-DD")
    parser.add_argument("--suffix",  default="",
                        help="Append to all output filenames, e.g. '_2015_2016'")
    parser.add_argument("--smoke",   action="store_true",
                        help="Smoke test: 10 mega-cap tickers, 6 months, separate cache.")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-download price data, ignoring any local cache.")
    args = parser.parse_args()

    if args.smoke:
        START_DATE = "2024-01-01"
        END_DATE   = "2024-06-30"
        ENTRY_DATE = "2024-01-02"
        suffix = args.suffix or "_smoke"
        CACHE_PATH = OUTPUT_DIR / "smoke_cache.parquet"
        tickers = SMOKE_TICKERS
        print(f"SMOKE MODE: {len(tickers)} tickers, {START_DATE} -> {END_DATE}")
    else:
        START_DATE = args.start
        END_DATE   = args.end
        ENTRY_DATE = args.entry
        suffix = args.suffix
        tickers = get_sp500_tickers()

    TRADE_LOG    = OUTPUT_DIR / f"trade_log{suffix}.csv"
    METRICS_TXT  = OUTPUT_DIR / f"metrics_summary{suffix}.txt"
    METRICS_CSV  = OUTPUT_DIR / f"metrics_summary{suffix}.csv"
    EQUITY_CSV   = OUTPUT_DIR / f"equity_curve{suffix}.csv"
    EQUITY_PNG   = OUTPUT_DIR / f"equity_curve{suffix}.png"
    DRAWDOWN_PNG = OUTPUT_DIR / f"drawdown{suffix}.png"

    prices = load_prices(tickers, refresh=args.refresh)

    # Restrict cache to configured window.
    mask = (prices.index >= pd.Timestamp(START_DATE)) & (prices.index <= pd.Timestamp(END_DATE))
    prices = prices.loc[mask]
    if len(prices) == 0:
        raise RuntimeError(f"No price data in window {START_DATE} -> {END_DATE}")

    trade_log, strat_eq, spy_eq, state = run_backtest(prices)
    write_outputs(trade_log, strat_eq, spy_eq, state)


if __name__ == "__main__":
    main()
