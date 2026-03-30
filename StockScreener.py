from __future__ import annotations
import sys
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

from Consensus import optimize_sleeves


warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.insert(0, '.')

from Factors import (
    momentum_12m_1m, momentum_6m,
    value_btm, value_pe_ratio, value_ev_ebitda, value_pb_ratio,
    quality_roe, quality_profit_margin, quality_sharpe, quality_earnings_stability,
    low_volatility_total,
    size_smb_tilt,
    composite_rank, generate_signals,
    compute_betas,   
)

#...................................
# CONFIG
#...................................

UNIVERSE_PATH  = str("universe.csv")
PRICES_CACHE   = str("prices_adj.parquet")
SIGNALS_OUTPUT = str("screener_signals.csv")
START_DATE     = "2020-01-01"
END_DATE       = "2026-01-01"
LOOKBACK       = 756
MAX_WORKERS    = 10

LIVE_SIGNALS_ONLY = True

SECTOR_WEIGHTS = {
    "Technology":             {'M': 0.35, 'V': 0.05, 'Q': 0.40, 'LV': 0.10, 'S': 0.10},
    "Healthcare":             {'M': 0.15, 'V': 0.20, 'Q': 0.50, 'LV': 0.10, 'S': 0.05},
    "Financial Services":     {'M': 0.10, 'V': 0.45, 'Q': 0.30, 'LV': 0.10, 'S': 0.05},
    "Consumer Cyclical":      {'M': 0.30, 'V': 0.25, 'Q': 0.25, 'LV': 0.10, 'S': 0.10},
    "Communication Services": {'M': 0.20, 'V': 0.30, 'Q': 0.35, 'LV': 0.10, 'S': 0.05},
    "Industrials":            {'M': 0.30, 'V': 0.15, 'Q': 0.30, 'LV': 0.10, 'S': 0.15},
    "Consumer Defensive":     {'M': 0.10, 'V': 0.15, 'Q': 0.35, 'LV': 0.35, 'S': 0.05},
    "Energy":                 {'M': 0.20, 'V': 0.45, 'Q': 0.25, 'LV': 0.05, 'S': 0.05},
    "Utilities":              {'M': 0.05, 'V': 0.25, 'Q': 0.35, 'LV': 0.30, 'S': 0.05},
    "Real Estate":            {'M': 0.10, 'V': 0.25, 'Q': 0.30, 'LV': 0.30, 'S': 0.05},
    "Materials":              {'M': 0.25, 'V': 0.40, 'Q': 0.25, 'LV': 0.05, 'S': 0.05},
}

DEFAULT_WEIGHTS = {'M': 0.20, 'V': 0.30, 'Q': 0.30, 'LV': 0.15, 'S': 0.05}


#...................................
# STEP 1 — LOAD UNIVERSE
#...................................
def load_universe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

#...................................
# PRICES
#...................................
def load_prices(tickers: list[str]) -> pd.DataFrame:
    cache = Path(PRICES_CACHE)

    if cache.exists():
        prices = pd.read_parquet(cache)
        prices.columns = [str(c).upper() for c in prices.columns]
        missing = [t for t in tickers if t not in prices.columns]
        if len(prices) > 100 and not missing:
            print(f"[prices] Loaded from cache: {prices.shape}")
            return prices
        else:
            print("[prices] Cache stale or incomplete — re-downloading...")
            cache.unlink()

    print("[prices] Downloading...")
    raw = yf.download(tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)

    if raw.empty or len(raw) < 100:
        raise RuntimeError("[prices] Download empty — likely rate limited. Wait 5-10 min.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw['Close'].copy()
    else:
        raw = raw[['Close']] if 'Close' in raw.columns else raw.copy()

    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    raw.columns = [str(c).upper() for c in raw.columns]

    min_rows = int(len(raw) * 0.75)
    valid_cols = [c for c in raw.columns if raw[c].notna().sum() >= min_rows]
    dropped = sorted(set(raw.columns) - set(valid_cols))
    if dropped:
        print(f"[prices] Dropped {len(dropped)} tickers with insufficient history: {dropped}")

    raw = raw[valid_cols].loc[:, ~pd.Index(valid_cols).duplicated()]
    raw.to_parquet(PRICES_CACHE)
    print(f"[prices] Saved: {raw.shape}")
    return raw


#...................................
# SPY BENCHMARK
#...................................
def load_spy_returns(index: pd.DatetimeIndex) -> pd.Series:
    print("[benchmark] Downloading SPY...")
    spy = yf.download("SPY", start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)['Close']
    spy_ret = spy.squeeze().pct_change()
    spy_ret = spy_ret.reindex(index).fillna(0)
    print(f"[benchmark] SPY loaded: {len(spy_ret)} days")
    return spy_ret

#...................................
# FUNDAMENTALS
#...................................
_FUND_KEYS = {
    'bookValue':         'book_value_per_share',
    'sharesOutstanding': 'shares',
    'marketCap':         'market_cap',
    'forwardEps':        'eps',
    'netIncomeToCommon': 'net_income',
    'totalRevenue':      'revenue',
    'totalDebt':         'debt',
    'totalCash':         'cash',
    'ebitda':            'ebitda',
}

def _fetch_one(tkr: str) -> dict:
    try:
        info = yf.Ticker(tkr).info
        return {v: info.get(k, np.nan) for k, v in _FUND_KEYS.items()}
    except Exception:
        return {v: np.nan for v in _FUND_KEYS.values()}

def refresh_yf_session():
    try:
        _ = yf.Ticker("AAPL").info
        print("[session] yfinance authenticated")
    except Exception as e:
        print(f"[session] Warning: {e}")

def fetch_fundamentals(tickers: list[str], index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    print(f"[fundamentals] Fetching {len(tickers)} tickers with {MAX_WORKERS} threads...")
    print(f"[fundamentals] ⚠ WARNING: Using point-in-time fundamentals (live signals only — not backtest-safe)")
    raw: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        for future in as_completed(futures):
            tkr = futures[future]
            raw[tkr] = future.result()

    fund_dfs: dict[str, pd.DataFrame] = {}
    for field in _FUND_KEYS.values():
        series = {tkr: raw[tkr].get(field, np.nan) for tkr in tickers}
        fund_dfs[field] = pd.DataFrame(series, index=index).ffill()

    shares = fund_dfs['shares']
    fund_dfs['book_equity'] = fund_dfs['book_value_per_share'] * shares
    fund_dfs['earnings']    = fund_dfs['eps'] * shares
    fund_dfs['equity']      = fund_dfs['book_equity']

    print(f"[fundamentals] Done. Fields: {list(fund_dfs.keys())}")
    return fund_dfs


#...................................
# FACTOR COMPUTATION
#...................................
def compute_factors(prices: pd.DataFrame, returns: pd.DataFrame,
                    funds: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:

    def avg(*dfs: pd.DataFrame) -> pd.DataFrame:
        stacked = np.nanmean([df.values for df in dfs], axis=0)
        return pd.DataFrame(stacked, index=dfs[0].index, columns=dfs[0].columns)

    M = avg(momentum_12m_1m(prices), momentum_6m(prices))

    V = avg(
        value_btm(funds['book_equity'], funds['market_cap']),
        value_pe_ratio(prices, funds['earnings']),
        value_ev_ebitda(funds['market_cap'], funds['debt'], funds['cash'], funds['ebitda']),
        value_pb_ratio(prices, funds['book_value_per_share']),
    )

    Q = avg(
        quality_roe(funds['net_income'], funds['equity']),
        quality_profit_margin(funds['net_income'], funds['revenue']),
        quality_sharpe(returns),
        quality_earnings_stability(funds['earnings']),
    )

    LV = low_volatility_total(returns)
    S  = size_smb_tilt(funds['market_cap'])

    return {'M': M, 'V': V, 'Q': Q, 'LV': LV, 'S': S}

#...................................
# BETA-NEUTRAL WEIGHT ADJUSTMENT
#...................................
def beta_neutral_weights(results_df: pd.DataFrame, betas: pd.Series) -> pd.DataFrame:
    df = results_df.copy()
    df['beta'] = df['ticker'].map(betas).fillna(1.0)

    for sector in df['sector'].unique():
        lmask = (df['sector'] == sector) & (df['signal'] ==  1)
        smask = (df['sector'] == sector) & (df['signal'] == -1)

        if df[lmask].empty or df[smask].empty:
            continue

        for mask in [lmask, smask]:
            sub      = df[mask]
            inv_beta = 1.0 / sub['beta'].clip(lower=0.2)
            norm_w   = inv_beta / inv_beta.sum()
            df.loc[mask, 'weight'] = norm_w.values * df.loc[mask, 'signal'].values

        beta_long  = (df.loc[lmask, 'weight'] * df.loc[lmask, 'beta']).sum()
        beta_short = (df.loc[smask, 'weight'] * df.loc[smask, 'beta']).sum()

        if abs(beta_short) > 1e-6:
            k = -beta_long / beta_short
            k = np.clip(k, 0.5, 2.0)
            df.loc[smask, 'weight'] *= k

    return df


#...................................
# BETA EXPOSURE REPORT
#...................................
def report_beta_exposure(results: pd.DataFrame, betas: pd.Series) -> None:
    df = results.copy()
    df['beta'] = df['ticker'].map(betas).fillna(1.0)

    print(f"\n{'='*55}")
    print(f"{'BETA EXPOSURE BY SECTOR':^55}")
    print(f"{'='*55}")
    print(f"  {'Sector':<23} {'Long β':>8} {'Short β':>9} {'Net β':>8}")
    print(f"  {'-'*51}")

    total_net = 0.0
    for sector in df['sector'].unique():
        s      = df[df['sector'] == sector]
        b_long = (s[s['signal'] ==  1]['weight']       * s[s['signal'] ==  1]['beta']).sum()
        b_short= (s[s['signal'] == -1]['weight'].abs() * s[s['signal'] == -1]['beta']).sum()
        net    = b_long - b_short
        total_net += net
        flag   = "HIGH" if abs(net) > 0.10 else ""
        print(f"  {sector:<23} {b_long:>8.3f} {b_short:>9.3f} {net:>8.3f}{flag}")

    print(f"  {'-'*51}")
    print(f"  {'TOTAL PORTFOLIO':<23} {'':>8} {'':>9} {total_net:>8.3f}")
    print(f"{'='*55}\n")

#...................................
# SECTOR SLEEVE SCREENER
#...................................
def screen_sector_sleeves(
    universe_df: pd.DataFrame,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    funds: dict[str, pd.DataFrame],
    betas: pd.Series,
    long_pct: float = 0.20,
    short_pct: float = 0.20,
) -> pd.DataFrame:
    all_results = []
    sectors = universe_df['sector'].unique()
    print(f"\n[sleeves] Building {len(sectors)} sector sleeves...")

    for sector in sectors:
        sector_tickers = universe_df[universe_df['sector'] == sector]['ticker'].tolist()
        sector_tickers = [t for t in sector_tickers if t in prices.columns]

        if len(sector_tickers) < 5:
            print(f"  [{sector}] only {len(sector_tickers)} tickers — skipping")
            continue

        p = prices[sector_tickers]
        r = returns[sector_tickers]
        f = {k: v[sector_tickers] for k, v in funds.items()}

        factors  = compute_factors(p, r, f)
        weights  = SECTOR_WEIGHTS.get(sector, DEFAULT_WEIGHTS)
        active_w = {k: weights[k] for k in factors if k in weights}
        total    = sum(active_w.values())
        active_w = {k: v / total for k, v in active_w.items()}

        composite = composite_rank(factors, active_w, universe_df=universe_df)
        signals   = generate_signals(composite,
                                     long_percentile=1 - long_pct,
                                     short_percentile=short_pct)

        latest_scores  = composite.iloc[-1]
        latest_signals = signals.iloc[-1]
        n_long  = int(len(sector_tickers) * long_pct)
        n_short = int(len(sector_tickers) * short_pct)

        for tkr in sector_tickers:
            score  = latest_scores.get(tkr, np.nan)
            signal = latest_signals.get(tkr, 0)
            if pd.isna(score):
                continue
            all_results.append({
                'sector':   sector,
                'ticker':   tkr,
                'score':    round(float(score), 4),
                'signal':   int(signal),
                'position': 'LONG' if signal == 1 else ('SHORT' if signal == -1 else 'NEUTRAL'),
                'weight':   0.0,
                'beta':     float(betas.get(tkr, 1.0)),
            })

        print(f"  [{sector}] {len(sector_tickers)} tickers → "
              f"{n_long} longs, {n_short} shorts")

    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        print("[sleeves] WARNING: No results — check fundamentals and price data")
        return results_df

    for sector in results_df['sector'].unique():
        mask_long  = (results_df['sector'] == sector) & (results_df['signal'] == 1)
        mask_short = (results_df['sector'] == sector) & (results_df['signal'] == -1)
        n_long     = mask_long.sum()
        n_short    = mask_short.sum()
        if n_long  > 0:
            results_df.loc[mask_long,  'weight'] =  round(1.0 / n_long,  4)
        if n_short > 0:
            results_df.loc[mask_short, 'weight'] = -round(1.0 / n_short, 4)

    return results_df.sort_values(['sector', 'signal'], ascending=[True, False])


#...................................
# MAIN
#...................................
def run_screener() -> pd.DataFrame:
    universe_df = load_universe(UNIVERSE_PATH)
    tickers     = universe_df['ticker'].tolist()

    prices  = load_prices(tickers)
    tickers = prices.columns.tolist()

    prices_t  = prices.tail(LOOKBACK)
    returns_t = prices_t.pct_change()

    spy_returns = load_spy_returns(prices_t.index)
    betas = compute_betas(returns_t, spy_returns)
    print(f"[betas] Computed for {len(betas)} tickers. "
          f"Mean: {betas.mean():.3f}, Min: {betas.min():.3f}, Max: {betas.max():.3f}")

    refresh_yf_session()
    funds = fetch_fundamentals(tickers, prices_t.index)
    funds = {k: v.reindex(columns=tickers) for k, v in funds.items()}

    universe_df = universe_df[universe_df['ticker'].isin(tickers)]

    results = screen_sector_sleeves(universe_df, prices_t, returns_t, funds, betas)

    results = optimize_sleeves(results, betas=betas, verbose=True)

    results.to_csv(SIGNALS_OUTPUT, index=False)

    if results.empty:
        print("No signals generated — fundamentals may still be all NaN")
        return results

    report_beta_exposure(results, betas)

    print(f"\n{'='*55}")
    print(f"{'SECTOR SLEEVE SUMMARY':^55}")
    print(f"{'='*55}")

    for sector in results['sector'].unique():
        s   = results[results['sector'] == sector]
        lng = s[s['signal'] == 1]['ticker'].tolist()
        sht = s[s['signal'] == -1]['ticker'].tolist()
        print(f"\n  {sector}")
        print(f"    Longs  ({len(lng)}): {lng}")
        print(f"    Shorts ({len(sht)}): {sht}")

    print(f"\n{'='*55}")
    print(f"Total longs : {(results['signal'] == 1).sum()}")
    print(f"Total shorts: {(results['signal'] == -1).sum()}")
    print(f"Output      : {SIGNALS_OUTPUT}")
    print(f"{'='*55}\n")

    return results


if __name__ == "__main__":
    run_screener()
