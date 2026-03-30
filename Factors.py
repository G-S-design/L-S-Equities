from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mstats


# ═══════════════════════════════════════════════════════════════
# MOMENTUM
# ═══════════════════════════════════════════════════════════════

def momentum_12m_1m(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.shift(21) / prices.shift(252) - 1


def momentum_6m(prices: pd.DataFrame) -> pd.DataFrame:
    return prices / prices.shift(126) - 1


# ═══════════════════════════════════════════════════════════════
# VALUE
# ═══════════════════════════════════════════════════════════════

def value_btm(book_equity: pd.DataFrame, market_cap: pd.DataFrame) -> pd.DataFrame:
    return book_equity / market_cap.replace(0, np.nan)


def value_pe_ratio(prices: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    pe = prices / earnings
    return 1.0 / pe.replace([np.inf, -np.inf], np.nan)


def value_ev_ebitda(
    market_cap: pd.DataFrame,
    debt:       pd.DataFrame,
    cash:       pd.DataFrame,
    ebitda:     pd.DataFrame,
) -> pd.DataFrame:
    ev       = market_cap + debt - cash
    ev_ebitda = ev / ebitda
    return 1.0 / ev_ebitda.replace([np.inf, -np.inf], np.nan)


def value_pb_ratio(prices: pd.DataFrame, book_value: pd.DataFrame) -> pd.DataFrame:
    pb = prices / book_value
    return 1.0 / pb.replace([np.inf, -np.inf], np.nan)


# ═══════════════════════════════════════════════════════════════
# QUALITY
# ═══════════════════════════════════════════════════════════════

def quality_roe(net_income: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    return net_income / equity.replace(0, np.nan)


def quality_profit_margin(net_income: pd.DataFrame, revenue: pd.DataFrame) -> pd.DataFrame:
    return net_income / revenue.replace(0, np.nan)


def quality_sharpe(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    mu    = returns.rolling(window).mean()
    sigma = returns.rolling(window).std().replace(0, np.nan)
    return mu / sigma


def quality_earnings_stability(earnings: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    
    mean_e = earnings.rolling(window).mean()
    std_e  = earnings.rolling(window).std()
    cv     = std_e / mean_e.replace(0, np.nan)
    return 1.0 / cv.replace([np.inf, -np.inf], np.nan)


# ═══════════════════════════════════════════════════════════════
# LOW VOLATILITY
# ═══════════════════════════════════════════════════════════════

def low_volatility_total(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
  
    vol = returns.rolling(window).std()
    return 1.0 / vol.replace(0, np.nan)


def low_volatility_idiosyncratic(
    returns:        pd.DataFrame,
    market_returns: pd.Series,
    window:         int = 60,
) -> pd.DataFrame:
   
    residuals = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for col in returns.columns:
        cov_    = returns[col].rolling(window).cov(market_returns)
        var_mkt = market_returns.rolling(window).var().replace(0, np.nan)
        beta    = cov_ / var_mkt
        residuals[col] = returns[col] - beta * market_returns

    ivol = residuals.rolling(window).std()
    return 1.0 / ivol.replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════
# SIZE  (Fama–French SMB tilt)
# ═══════════════════════════════════════════════════════════════

def size_log_market_cap(market_cap: pd.DataFrame) -> pd.DataFrame:
    
    return np.log(market_cap.replace(0, np.nan))


def size_smb_tilt(market_cap: pd.DataFrame) -> pd.DataFrame:
   
    return -np.log(market_cap.replace(0, np.nan))


# ═══════════════════════════════════════════════════════════════
# BETA COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_betas(
    returns:        pd.DataFrame,
    market_returns: pd.Series,
    window:         int = 252,
) -> pd.Series:

    betas       = {}
    market_tail = market_returns.reindex(returns.index).fillna(0)

    for tkr in returns.columns:
        r = returns[tkr].dropna().tail(window)
        m = market_tail.reindex(r.index).fillna(0)

        if len(r) < 60:
            betas[tkr] = 1.0
            continue

        r_arr = r.values
        m_arr = m.values
        m_dm  = m_arr - m_arr.mean()
        market_var = float(np.dot(m_dm, m_dm) / len(m_dm))

        if market_var < 1e-10:
            betas[tkr] = 1.0
        else:
            r_dm      = r_arr - r_arr.mean()
            cov_rm    = float(np.dot(r_dm, m_dm) / len(m_dm))
            betas[tkr] = cov_rm / market_var

    return pd.Series(betas, name='beta')


# ═══════════════════════════════════════════════════════════════
# NORMALISATION PIPELINE
# ═══════════════════════════════════════════════════════════════

def winsorize(factor: pd.DataFrame, limits: tuple = (0.05, 0.05)) -> pd.DataFrame:
    def _winsorize_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if len(valid) < 3:
            return row
        row[valid.index] = mstats.winsorize(valid.values, limits=limits)
        return row

    return factor.apply(_winsorize_row, axis=1)


def z_score_normalize(factor: pd.DataFrame) -> pd.DataFrame:
    mean = factor.mean(axis=1)
    std  = factor.std(axis=1).replace(0, np.nan)
    return factor.sub(mean, axis=0).div(std, axis=0)


def sector_zscore(
    composite:   pd.DataFrame,
    universe_df: pd.DataFrame,
) -> pd.DataFrame:
    out = composite.copy()

    for _, grp in universe_df.groupby('sector'):
        tickers = [t for t in grp['ticker'] if t in composite.columns]
        if len(tickers) < 2:
            continue
        sub  = composite[tickers]
        mean = sub.mean(axis=1)
        std  = sub.std(axis=1).clip(lower=1e-6)
        out[tickers] = sub.sub(mean, axis=0).div(std, axis=0)

    return out


def composite_rank(
    factors_dict: dict[str, pd.DataFrame],
    weights:      dict[str, float],
    normalize:    bool = True,
    universe_df:  pd.DataFrame | None = None,
) -> pd.DataFrame:
  
    first  = factors_dict[list(factors_dict.keys())[0]]
    composite = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    coverage  = pd.DataFrame(0.0, index=first.index, columns=first.columns)

    for name, factor_df in factors_dict.items():
        w = weights.get(name, 1.0)

        if normalize:
            factor_z = z_score_normalize(winsorize(factor_df))
        else:
            factor_z = factor_df

        valid = factor_z.notna()
        composite += (w * factor_z).fillna(0.0)
        coverage  += valid.astype(float) * w

    # Where no factor contributed at all, set to NaN
    composite[coverage == 0] = np.nan

    if universe_df is not None:
        composite = sector_zscore(composite, universe_df)

    return composite


def generate_signals(
    composite_scores: pd.DataFrame,
    long_percentile:  float = 0.80,
    short_percentile: float = 0.20,
) -> pd.DataFrame:
 
    ranks = composite_scores.rank(axis=1, pct=True)

    signals = pd.DataFrame(0, index=composite_scores.index, columns=composite_scores.columns)
    signals[ranks >= long_percentile]  =  1
    signals[ranks <= short_percentile] = -1

    return signals