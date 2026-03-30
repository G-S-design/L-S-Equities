"""
inspect_parquet.py
------------------
Quick sanity check on prices_adj.parquet before running the backtest.
Run from the same directory as your parquet file:

    python inspect_parquet.py
"""

import pandas as pd

PARQUET_PATH = "prices_adj.parquet"
SIGNALS_PATH = "screener_signals.csv"
SLEEVES_PATH = "sleeves.csv"
BACKTEST_START = "2023-01-01"   # must match what you want to backtest

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading parquet...")
prices = pd.read_parquet(PARQUET_PATH)

# Normalize — expect shape (dates x tickers), if transposed fix it
if prices.index.dtype == object and not isinstance(prices.index[0], str):
    pass
# If tickers are on columns, dates on index — that's correct
# If reversed, uncomment: prices = prices.T

print(f"\n{'='*55}")
print("  PARQUET OVERVIEW")
print(f"{'='*55}")
print(f"  Shape:         {prices.shape[0]} rows x {prices.shape[1]} columns")
print(f"  Date range:    {prices.index[0]}  →  {prices.index[-1]}")
print(f"  Total tickers: {prices.shape[1]}")
print(f"  Freq guess:    {'Daily' if len(prices) > 200 else 'Sub-daily or sparse'}")

# ── DATE COVERAGE CHECK ───────────────────────────────────────────────────────
start_ok = pd.Timestamp(BACKTEST_START)
actual_start = pd.Timestamp(prices.index[0])
actual_end   = pd.Timestamp(prices.index[-1])

print(f"\n── DATE COVERAGE ─────────────────────────────────────────────")
if actual_start <= start_ok:
    print(f"  ✅ Starts on/before {BACKTEST_START} — backtest start date is covered")
else:
    print(f"  ❌ Data starts {actual_start.date()}, AFTER requested backtest start {BACKTEST_START}")
    print(f"     → Either adjust BACKTEST_START in backtest.py or use a longer parquet")

today = pd.Timestamp("today").normalize()
days_stale = (today - actual_end).days
if days_stale <= 5:
    print(f"  ✅ Data is current (last date: {actual_end.date()}, {days_stale} days ago)")
else:
    print(f"  ⚠️  Data ends {actual_end.date()} — {days_stale} days stale")
    print(f"     → Backtest will only run to {actual_end.date()}")

# ── SCREENER TICKER COVERAGE ──────────────────────────────────────────────────
signals = pd.read_csv(SIGNALS_PATH)
sleeves = pd.read_csv(SLEEVES_PATH).set_index("sleeve")["sleeve_weight"]

EXCLUDED = {"Consumer Cyclical", "Materials", "Real Estate"}
active = signals[
    (signals["position"].isin(["LONG", "SHORT"])) &
    (~signals["sector"].isin(EXCLUDED))
].copy()

needed  = set(active["ticker"].tolist())
have    = set(prices.columns.tolist())
present = needed & have
missing = needed - have

print(f"\n── TICKER COVERAGE ───────────────────────────────────────────")
print(f"  Active positions needed:  {len(needed)}")
print(f"  Found in parquet:         {len(present)}  ✅")
print(f"  Missing from parquet:     {len(missing)}  {'✅ none' if not missing else '⚠️'}")
if missing:
    print(f"  Missing tickers: {sorted(missing)}")

# ── NaN / DATA QUALITY ────────────────────────────────────────────────────────
subset = prices[list(present)]
nan_pct = (subset.isna().sum() / len(subset) * 100).sort_values(ascending=False)
bad = nan_pct[nan_pct > 5]

print(f"\n── DATA QUALITY (active tickers only) ────────────────────────")
if bad.empty:
    print(f"  ✅ No tickers with >5% missing data")
else:
    print(f"  ⚠️  {len(bad)} tickers with >5% NaN (will be forward-filled):")
    for t, pct in bad.items():
        print(f"     {t:<12}  {pct:.1f}% missing")

zero_price = (subset <= 0).any()
bad_zeros = zero_price[zero_price].index.tolist()
if bad_zeros:
    print(f"  ⚠️  Zero/negative prices found in: {bad_zeros}")
else:
    print(f"  ✅ No zero or negative prices")

# ── SUMMARY VERDICT ───────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  VERDICT")
print(f"{'='*55}")
coverage_pct = len(present) / len(needed) * 100
if coverage_pct == 100 and actual_start <= start_ok and days_stale <= 5 and bad.empty:
    print("  ✅ Parquet looks good — ready to run backtest")
else:
    print(f"  Coverage:  {coverage_pct:.1f}% of active tickers found")
    if actual_start > start_ok:
        print(f"  ❌ Date range issue — parquet starts too late")
    if days_stale > 5:
        print(f"  ⚠️  Stale data — backtest will end {actual_end.date()}")
    if not bad.empty:
        print(f"  ⚠️  Some tickers have significant gaps (will ffill)")
    if missing:
        print(f"  ⚠️  {len(missing)} tickers missing — will be skipped in backtest")
print(f"{'='*55}\n")