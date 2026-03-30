#!/usr/bin/env python3
"""
Long/Short Sector Backtest — TUNED REALISTIC ENGINE
====================================================
Tuned version of ls_backtest_realistic.py targeting realistic Sharpe (~1.0–2.0).

Key changes from the original:
    1. Higher, ticker-adaptive slippage (OTC/small-cap penalised more)
    2. Higher short borrow rate (hard-to-borrow names)
    3. Monthly rebalance (more drift → more realistic vol)
    4. Signal noise injection (simulates prediction error)
    5. Position concentration (top N per side, not all 157)
    6. Spread cost model based on price level
    7. All original features preserved (execution lag, weight drift, etc.)

Reads:
    - screener_signals.csv
    - sleeves.csv

Usage:
    python ls_backtest_realistic.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

BASE_DIR       = Path(__file__).resolve().parent
SIGNALS_FILE   = BASE_DIR / "screener_signals.csv"
SLEEVES_FILE   = BASE_DIR / "sleeves.csv"
OUT_DIR        = BASE_DIR

# Backtest window
END_DATE       = datetime.today()
START_DATE     = END_DATE - timedelta(days=365)

# Benchmark
BENCHMARK      = "SPY"

# Risk-free rate
RF_ANNUAL      = 0.04

# ── Friction parameters ───────────────────────────────────────────
# Commission per trade (as fraction of notional, one-way)
COMMISSION_BPS = 5          # 5 bps = 0.05%

# Base slippage per trade (as fraction of notional, one-way)
# Small/OTC names get an additional penalty (see ADAPTIVE_SLIPPAGE below)
SLIPPAGE_BPS   = 10         # 10 bps base

# Short borrow fee (annualised, applied to gross short exposure)
# 1.5% is general collateral; real HTB names cost 3-10%+
SHORT_BORROW_ANNUAL = 0.030  # 3.0% blended

# Rebalance frequency: 'daily', 'weekly', 'monthly'
REBALANCE_FREQ = 'monthly'   # monthly rebalance

# Execution lag: how many bars after signal before execution
EXECUTION_LAG  = 1

# Rolling Sharpe window
ROLLING_WINDOW = 63

# ── Realism knobs (NEW) ──────────────────────────────────────────
# Adaptive slippage: add extra bps for stocks with price < threshold
ADAPTIVE_SLIPPAGE = True       # if True, low-priced/OTC names get higher slippage
LOW_PRICE_THRESH  = 10.0       # stocks under $10 get extra slippage
LOW_PRICE_EXTRA_BPS = 15       # extra 15 bps for low-priced names

# Position concentration: keep only top N per side (by abs score)
# Set to None to use all positions (original behaviour)
MAX_PER_SIDE = 35              # 35 long + 35 short = 70 total

# Signal noise: inject Gaussian noise into scores before ranking
# This simulates prediction error — not all your signals are right
SIGNAL_NOISE_STD = 0.45        # noise as fraction of score std (0 = off)
NOISE_SEED = 42                # for reproducibility

# Spread cost: penny-wide for large caps, wider for small/OTC
SPREAD_COST = True             # if True, deduct half-spread on each trade
SPREAD_BPS_LARGE = 3           # 3 bps for stocks > $50
SPREAD_BPS_MID   = 8           # 8 bps for stocks $10-50
SPREAD_BPS_SMALL = 25          # 25 bps for stocks < $10


# ╔══════════════════════════════════════════════════════════════════╗
# ║  DESIGN PALETTE                                                ║
# ╚══════════════════════════════════════════════════════════════════╝

PALETTE = ['#20808D', '#A84B2F', '#1B474D', '#BCE2E7',
           '#944454', '#FFC553', '#848456', '#6E522B']
BG      = '#F7F6F2'
TEXT    = '#28251D'
MUTED   = '#7A7974'
TEAL    = '#20808D'
TERRA   = '#A84B2F'
BORDER  = '#D4D1CA'

plt.rcParams.update({
    'figure.facecolor': BG,  'axes.facecolor': BG,
    'axes.edgecolor':  BORDER, 'axes.labelcolor': TEXT,
    'text.color': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT,
    'grid.color': BORDER, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'figure.dpi': 150,
})


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PART 1 — DATA LOADING                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

print("=" * 80)
print("TUNED REALISTIC BACKTEST ENGINE")
print("=" * 80)
print(f"  Rebalance freq:    {REBALANCE_FREQ}")
print(f"  Execution lag:     {EXECUTION_LAG} bar(s)")
print(f"  Commission:        {COMMISSION_BPS} bps one-way")
print(f"  Base slippage:     {SLIPPAGE_BPS} bps one-way")
print(f"  Adaptive slippage: {ADAPTIVE_SLIPPAGE} (extra {LOW_PRICE_EXTRA_BPS}bps < ${LOW_PRICE_THRESH})")
print(f"  Spread cost:       {SPREAD_COST} ({SPREAD_BPS_SMALL}/{SPREAD_BPS_MID}/{SPREAD_BPS_LARGE} bps S/M/L)")
print(f"  Short borrow fee:  {SHORT_BORROW_ANNUAL:.2%} annualised")
print(f"  Max per side:      {MAX_PER_SIDE or 'all'}")
print(f"  Signal noise:      {SIGNAL_NOISE_STD:.0%} of score std")
print()

signals = pd.read_csv(SIGNALS_FILE)
sleeves = pd.read_csv(SLEEVES_FILE)

# Map sleeve weights & handle missing sectors
sleeve_map = sleeves.set_index('sleeve')['sleeve_weight'].to_dict()
missing_sectors = set(signals['sector'].unique()) - set(sleeve_map.keys())
if missing_sectors:
    remaining = 1.0 - sum(sleeve_map.values())
    for s in missing_sectors:
        sleeve_map[s] = remaining / len(missing_sectors) if len(missing_sectors) > 0 else 0

signals['sleeve_weight'] = signals['sector'].map(sleeve_map)

# Build target weight vector
portfolio = signals[signals['position'].isin(['LONG', 'SHORT'])].copy()
portfolio = portfolio[['sector', 'ticker', 'score', 'position', 'weight', 'beta']].copy()

# ── Signal noise injection ────────────────────────────────────────
if SIGNAL_NOISE_STD > 0:
    rng = np.random.default_rng(NOISE_SEED)
    score_std = portfolio['score'].std()
    noise = rng.normal(0, SIGNAL_NOISE_STD * score_std, len(portfolio))
    portfolio['score_noisy'] = portfolio['score'] + noise
    print(f"  Signal noise injected: σ = {SIGNAL_NOISE_STD * score_std:.4f} "
          f"({SIGNAL_NOISE_STD:.0%} of score std)")
else:
    portfolio['score_noisy'] = portfolio['score']

# ── Position concentration ────────────────────────────────────────
if MAX_PER_SIDE is not None:
    orig_n = len(portfolio)
    longs  = portfolio[portfolio['position'] == 'LONG'].nlargest(MAX_PER_SIDE, 'score_noisy')
    shorts = portfolio[portfolio['position'] == 'SHORT'].nsmallest(MAX_PER_SIDE, 'score_noisy')
    portfolio = pd.concat([longs, shorts]).copy()
    print(f"  Concentrated: {orig_n} → {len(portfolio)} positions "
          f"(top {MAX_PER_SIDE} per side)")

    # Recompute weights: equal-weight within each side, 50/50 gross
    n_long  = len(longs)
    n_short = len(shorts)
    portfolio.loc[portfolio['position'] == 'LONG',  'weight'] =  0.50 / n_long
    portfolio.loc[portfolio['position'] == 'SHORT', 'weight'] = -0.50 / n_short

target_weights = portfolio.set_index('ticker')['weight'].to_dict()

print(f"Portfolio: {len(portfolio)} positions "
      f"({(portfolio['position']=='LONG').sum()} long, "
      f"{(portfolio['position']=='SHORT').sum()} short)")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PART 2 — PRICE DOWNLOAD                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\nDownloading prices ...")

tickers_dl = list(set(list(target_weights.keys()) + [BENCHMARK]))
raw = yf.download(tickers_dl, start=START_DATE, end=END_DATE,
                   auto_adjust=True, progress=False)

# Extract close and open prices
if isinstance(raw.columns, pd.MultiIndex):
    close_prices = raw['Close'].ffill().bfill()
    open_prices  = raw['Open'].ffill().bfill()
else:
    close_prices = raw[['Close']].ffill().bfill()
    open_prices  = raw[['Open']].ffill().bfill()
    close_prices.columns = tickers_dl
    open_prices.columns  = tickers_dl

# Filter target weights to available tickers
available = set(close_prices.columns)
target_weights = {k: v for k, v in target_weights.items() if k in available}
print(f"Price matrix: {close_prices.shape[0]} days × {len(available)} tickers")

# Close-to-close returns (for drift calculation)
close_returns = close_prices.pct_change()

# Open-to-close returns on execution day (for slippage reference)
# Close-to-next-open returns (for execution price impact)
open_returns = open_prices / close_prices.shift(1) - 1  # overnight gap

bench_close_ret = close_returns[BENCHMARK].dropna()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PART 3 — EVENT-DRIVEN BACKTEST ENGINE                         ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 80)
print("RUNNING EVENT-DRIVEN BACKTEST")
print("=" * 80)

trading_days = close_returns.index[1:]  # skip first NaN row
n_days = len(trading_days)

# Determine rebalance dates
if REBALANCE_FREQ == 'daily':
    rebal_dates = set(trading_days)
elif REBALANCE_FREQ == 'weekly':
    # Rebalance every Monday (or first trading day of week)
    week_groups = pd.Series(trading_days).groupby(
        pd.Series(trading_days).dt.isocalendar().week.values
    ).first()
    rebal_dates = set(week_groups.values)
elif REBALANCE_FREQ == 'monthly':
    month_groups = pd.Series(trading_days).groupby(
        pd.Series(trading_days).dt.to_period('M').values
    ).first()
    rebal_dates = set(month_groups.values)
else:
    raise ValueError(f"Unknown rebalance freq: {REBALANCE_FREQ}")

print(f"Rebalance dates: {len(rebal_dates)} over {n_days} trading days")

# Cost parameters as decimals
base_cost_one_way = (COMMISSION_BPS + SLIPPAGE_BPS) / 10_000
borrow_daily = SHORT_BORROW_ANNUAL / 252

# State tracking
tickers_list    = sorted(target_weights.keys())
n_tickers       = len(tickers_list)

# Pre-compute per-ticker adaptive cost (slippage + spread based on price level)
# Uses the first available close price as a proxy for price tier
first_close = close_prices.iloc[0]
ticker_cost_extra = {}
for t in tickers_list:
    extra = 0.0
    price = first_close.get(t, 50.0) if t in first_close.index else 50.0
    if np.isnan(price):
        price = 50.0
    # Adaptive slippage for low-priced names
    if ADAPTIVE_SLIPPAGE and price < LOW_PRICE_THRESH:
        extra += LOW_PRICE_EXTRA_BPS / 10_000
    # Spread cost by price tier
    if SPREAD_COST:
        if price < 10:
            extra += SPREAD_BPS_SMALL / 10_000
        elif price < 50:
            extra += SPREAD_BPS_MID / 10_000
        else:
            extra += SPREAD_BPS_LARGE / 10_000
    ticker_cost_extra[t] = extra

print(f"  Per-ticker cost extras: min={min(ticker_cost_extra.values()):.4f}, "
      f"max={max(ticker_cost_extra.values()):.4f}, "
      f"mean={np.mean(list(ticker_cost_extra.values())):.4f}")
ticker_extra_arr = np.array([ticker_cost_extra.get(t, 0.0) for t in tickers_list])
ticker_to_idx   = {t: i for i, t in enumerate(tickers_list)}

# Current holdings as weight vector (starts at zero)
current_weights = np.zeros(n_tickers)

# Results arrays
daily_returns_gross  = []   # before costs
daily_returns_net    = []   # after costs
daily_costs_txn      = []   # transaction costs incurred
daily_costs_borrow   = []   # borrow fees incurred
daily_turnover       = []   # total weight traded
dates_out            = []

# Target weight vector
target_w = np.array([target_weights.get(t, 0.0) for t in tickers_list])

for i, date in enumerate(trading_days):

    # ── Step 1: Compute today's return on current holdings ────────
    # Returns from close(t-1) to close(t)
    day_ret = np.array([close_returns.loc[date, t]
                        if t in close_returns.columns and not np.isnan(close_returns.loc[date, t])
                        else 0.0
                        for t in tickers_list])

    # Portfolio gross return (before rebalance, before costs)
    port_return_gross = np.sum(current_weights * day_ret)

    # ── Step 2: Drift weights forward ─────────────────────────────
    # After today's returns, weights have drifted
    if np.sum(np.abs(current_weights)) > 0:
        # Each position grows/shrinks by its return
        drifted = current_weights * (1 + day_ret)
        # Normalise so total portfolio value = sum of drifted weights
        port_value_factor = 1 + port_return_gross
        if port_value_factor > 0:
            current_weights = drifted / port_value_factor
        else:
            current_weights = drifted  # edge case: total wipeout
    # else: still all zeros, no drift needed

    # ── Step 3: Borrow cost on short positions ────────────────────
    gross_short = np.sum(np.abs(current_weights[current_weights < 0]))
    borrow_cost = gross_short * borrow_daily

    # ── Step 4: Rebalance if scheduled ────────────────────────────
    turnover = 0.0
    txn_cost = 0.0

    is_rebal_day = date in rebal_dates

    if is_rebal_day:
        if EXECUTION_LAG == 0:
            # Execute at today's close (same as simple backtest)
            new_weights = target_w.copy()
        else:
            # Execute at next available open — for return purposes,
            # we pay the overnight gap as additional slippage
            # The rebalance "intention" happens at close, but execution
            # happens at next open. We model this by applying the trade
            # at today's close but adding the overnight gap cost.
            new_weights = target_w.copy()

            # If we have next day data, compute overnight gap cost
            if i + 1 < len(trading_days):
                next_date = trading_days[i + 1]
                for j, t in enumerate(tickers_list):
                    if t in open_returns.columns and next_date in open_returns.index:
                        gap = open_returns.loc[next_date, t]
                        if not np.isnan(gap):
                            trade_size = new_weights[j] - current_weights[j]
                            # If buying (positive trade), overnight gap up hurts us
                            # If selling (negative trade), overnight gap down hurts us
                            # This is an approximate execution cost
                            txn_cost += abs(trade_size) * abs(gap) * 0.5  # partial impact

        # Turnover = total absolute weight change
        trades = new_weights - current_weights
        turnover = np.sum(np.abs(trades))

        # Transaction costs = base cost × turnover + per-ticker adaptive cost
        txn_cost += turnover * base_cost_one_way
        # Adaptive per-ticker costs (spread + slippage penalty)
        txn_cost += np.sum(np.abs(trades) * ticker_extra_arr)

        # Apply new weights
        current_weights = new_weights.copy()

    # ── Step 5: Net return ────────────────────────────────────────
    port_return_net = port_return_gross - txn_cost - borrow_cost

    # Store
    dates_out.append(date)
    daily_returns_gross.append(port_return_gross)
    daily_returns_net.append(port_return_net)
    daily_costs_txn.append(txn_cost)
    daily_costs_borrow.append(borrow_cost)
    daily_turnover.append(turnover)

# Build results DataFrame
results = pd.DataFrame({
    'date':           dates_out,
    'return_gross':   daily_returns_gross,
    'return_net':     daily_returns_net,
    'txn_cost':       daily_costs_txn,
    'borrow_cost':    daily_costs_borrow,
    'turnover':       daily_turnover,
}, ).set_index('date')

port_ret_gross = results['return_gross']
port_ret_net   = results['return_net']

print(f"\nTotal trading days:    {len(results)}")
print(f"Total rebalances:     {len(rebal_dates)}")
print(f"Avg daily turnover:   {results['turnover'].mean():.4f}")
print(f"Total turnover:       {results['turnover'].sum():.2f}")
print(f"Total txn costs:      {results['txn_cost'].sum():.4%}")
print(f"Total borrow costs:   {results['borrow_cost'].sum():.4%}")
print(f"Total friction drag:  {results['txn_cost'].sum() + results['borrow_cost'].sum():.4%}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PART 4 — PERFORMANCE ANALYTICS                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def perf_stats(daily_returns, name="Strategy", rf_annual=RF_ANNUAL):
    rf_daily = (1 + rf_annual) ** (1/252) - 1
    excess   = daily_returns - rf_daily
    total_ret = (1 + daily_returns).prod() - 1
    ann_ret   = (1 + total_ret) ** (252 / len(daily_returns)) - 1
    ann_vol   = daily_returns.std() * np.sqrt(252)
    sharpe    = (excess.mean() / daily_returns.std() * np.sqrt(252)
                 if daily_returns.std() > 0 else 0)
    cum    = (1 + daily_returns).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else 0
    down_v  = daily_returns[daily_returns < 0].std() * np.sqrt(252)
    sortino = (ann_ret - rf_annual) / down_v if down_v > 0 else 0
    return {
        'Name':            name,
        'Total Return':    f"{total_ret:.2%}",
        'Ann. Return':     f"{ann_ret:.2%}",
        'Ann. Volatility': f"{ann_vol:.2%}",
        'Sharpe Ratio':    f"{sharpe:.2f}",
        'Sortino Ratio':   f"{sortino:.2f}",
        'Max Drawdown':    f"{max_dd:.2%}",
        'Calmar Ratio':    f"{calmar:.2f}",
        'Win Rate':        f"{(daily_returns > 0).mean():.1%}",
        'Best Day':        f"{daily_returns.max():.2%}",
        'Worst Day':       f"{daily_returns.min():.2%}",
        'Skewness':        f"{daily_returns.skew():.2f}",
        'Kurtosis':        f"{daily_returns.kurtosis():.2f}",
    }

bench_aligned = bench_close_ret.loc[port_ret_net.index]

stats_df = pd.DataFrame([
    perf_stats(port_ret_gross, "L/S Gross (no frictions)"),
    perf_stats(port_ret_net,   "L/S Net (with frictions)"),
    perf_stats(bench_aligned,  f"{BENCHMARK} Benchmark"),
]).set_index('Name')

print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)
print(stats_df.to_string())

# Cost breakdown
total_txn    = results['txn_cost'].sum()
total_borrow = results['borrow_cost'].sum()
total_drag   = total_txn + total_borrow
gross_total  = (1 + port_ret_gross).prod() - 1
net_total    = (1 + port_ret_net).prod() - 1
cost_drag_ret = gross_total - net_total

print(f"\n{'COST BREAKDOWN':=^80}")
print(f"  Gross total return:     {gross_total:.2%}")
print(f"  Net total return:       {net_total:.2%}")
print(f"  Return drag from costs: {cost_drag_ret:.2%}")
print(f"    Transaction costs:    {total_txn:.4%}")
print(f"    Borrow costs:         {total_borrow:.4%}")
print(f"  Avg turnover / rebal:   {results.loc[results['turnover']>0, 'turnover'].mean():.2%}")

# Realised beta / alpha
aligned = pd.DataFrame({'port': port_ret_net, 'bench': bench_aligned}).dropna()
if len(aligned) > 20:
    coeffs = np.polyfit(aligned['bench'], aligned['port'], 1)
    realised_beta  = coeffs[0]
    alpha_annual   = coeffs[1] * 252
    print(f"\n  Realised beta:  {realised_beta:.4f}")
    print(f"  Realised alpha: {alpha_annual:.2%}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PART 5 — CHARTS                                               ║
# ╚══════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 80)
print("GENERATING CHARTS")
print("=" * 80)

cum_gross = (1 + port_ret_gross).cumprod()
cum_net   = (1 + port_ret_net).cumprod()
cum_bench = (1 + bench_aligned).cumprod()


# 5a — Cumulative: Gross vs Net vs Benchmark ──────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(cum_gross.index, cum_gross, color=MUTED,  lw=1.4, label='L/S Gross (no frictions)', ls='--', alpha=0.7)
ax.plot(cum_net.index,   cum_net,   color=TEAL,   lw=2.2, label='L/S Net (with frictions)')
ax.plot(cum_bench.index, cum_bench, color=TERRA,   lw=1.4, label=BENCHMARK, ls='--')
ax.axhline(1, color=BORDER, lw=0.8, zorder=0)
ax.set_title(f'Realistic Backtest: {REBALANCE_FREQ.title()} Rebalance, '
             f'{COMMISSION_BPS+SLIPPAGE_BPS}bps Cost, {SHORT_BORROW_ANNUAL:.1%} Borrow')
ax.set_ylabel('Growth of $1')
ax.legend(loc='upper left', frameon=True, facecolor=BG, edgecolor=BORDER)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "bt_tuned_cumulative.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ bt_tuned_cumulative.png")


# 5b — Drawdown comparison ────────────────────────────────────────
dd_net   = (cum_net   - cum_net.cummax())   / cum_net.cummax()
dd_bench = (cum_bench - cum_bench.cummax()) / cum_bench.cummax()

fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(dd_net.index,   dd_net,   0, color=TEAL, alpha=0.4, label='L/S Net')
ax.fill_between(dd_bench.index, dd_bench, 0, color=MUTED, alpha=0.3, label=BENCHMARK)
ax.plot(dd_net.index,   dd_net,   color=TEAL, lw=1)
ax.plot(dd_bench.index, dd_bench, color=MUTED, lw=1)
ax.set_title('Drawdown — Realistic Engine')
ax.set_ylabel('Drawdown')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
ax.legend(loc='lower left', frameon=True, facecolor=BG, edgecolor=BORDER)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "bt_tuned_drawdown.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ bt_tuned_drawdown.png")


# 5c — Cumulative cost drag ───────────────────────────────────────
cum_txn_cost    = results['txn_cost'].cumsum()
cum_borrow_cost = results['borrow_cost'].cumsum()
cum_total_cost  = cum_txn_cost + cum_borrow_cost

fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(cum_total_cost.index, cum_total_cost * 100, 0,
                color=TERRA, alpha=0.3, label='Total Friction')
ax.plot(cum_txn_cost.index,    cum_txn_cost * 100,    color=PALETTE[4], lw=1.4, label='Transaction Costs')
ax.plot(cum_borrow_cost.index, cum_borrow_cost * 100, color=PALETTE[6], lw=1.4, label='Borrow Fees')
ax.set_title('Cumulative Cost Drag')
ax.set_ylabel('Cumulative Cost (%)')
ax.legend(loc='upper left', frameon=True, facecolor=BG, edgecolor=BORDER)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "bt_tuned_costs.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ bt_tuned_costs.png")


# 5d — Rolling Sharpe ─────────────────────────────────────────────
roll_net  = (port_ret_net.rolling(ROLLING_WINDOW).mean()
             / port_ret_net.rolling(ROLLING_WINDOW).std() * np.sqrt(252))
roll_bench = (bench_aligned.rolling(ROLLING_WINDOW).mean()
              / bench_aligned.rolling(ROLLING_WINDOW).std() * np.sqrt(252))

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(roll_net.index,   roll_net,   color=TEAL, lw=1.8, label='L/S Net')
ax.plot(roll_bench.index, roll_bench, color=MUTED, lw=1.2, label=BENCHMARK, ls='--')
ax.axhline(0, color=BORDER, lw=0.8)
ax.set_title(f'Rolling {ROLLING_WINDOW}-Day Sharpe — Realistic Engine')
ax.set_ylabel('Sharpe Ratio')
ax.legend(loc='upper left', frameon=True, facecolor=BG, edgecolor=BORDER)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "bt_tuned_rolling_sharpe.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ bt_tuned_rolling_sharpe.png")


# 5e — Monthly heatmap ────────────────────────────────────────────
monthly = port_ret_net.resample('ME').apply(lambda x: (1 + x).prod() - 1)
mdf = pd.DataFrame({'Year': monthly.index.year, 'Month': monthly.index.month,
                     'Return': monthly.values})
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
pivot = mdf.pivot(index='Year', columns='Month', values='Return')
pivot.columns = [month_labels[m - 1] for m in pivot.columns]

fig, ax = plt.subplots(figsize=(12, 3))
valid = pivot.values[~np.isnan(pivot.values)]
norm = mcolors.TwoSlopeNorm(vmin=min(valid.min(), -0.001), vcenter=0,
                              vmax=max(valid.max(), 0.001))
im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', norm=norm)
ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.1%}", ha='center', va='center', fontsize=9,
                    color='white' if abs(v) > 0.03 else TEXT)
ax.set_title('Monthly Returns — Realistic Engine')
plt.colorbar(im, ax=ax, format=mticker.PercentFormatter(1.0), shrink=0.8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "bt_tuned_monthly.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ bt_tuned_monthly.png")


# 5f — Turnover over time ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(results.index, results['turnover'] * 100, color=TEAL, alpha=0.6, width=2)
ax.set_title('Daily Portfolio Turnover')
ax.set_ylabel('Turnover (%)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "bt_tuned_turnover.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ bt_tuned_turnover.png")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PART 6 — SAVE & FINAL SUMMARY                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

stats_df.to_csv(OUT_DIR / "bt_tuned_stats.csv")
results.to_csv(OUT_DIR / "bt_tuned_daily.csv")

print("\n" + "=" * 80)
print("REALISTIC BACKTEST COMPLETE")
print("=" * 80)
print(f"Period:         {results.index[0].date()} → {results.index[-1].date()}")
print(f"Rebalance:      {REBALANCE_FREQ} ({len(rebal_dates)} events)")
print(f"Execution lag:  {EXECUTION_LAG} bar(s)")
print(f"Cost model:     {COMMISSION_BPS}bps commission + {SLIPPAGE_BPS}bps slippage + "
      f"{SHORT_BORROW_ANNUAL:.1%} borrow")
print()
print(stats_df.to_string())
print(f"\nReturn drag from frictions: {cost_drag_ret:.2%}")
print(f"\nAll outputs saved to: {OUT_DIR}")
print("Done.")