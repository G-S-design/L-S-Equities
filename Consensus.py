from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

NSTEPS          = 400        # Consensus iterations
EPSILON         = 0.05      
STEP_SIZE       = 0.04      
BASE_MU         = 1.0       

# Risk controls
RISK_AVERSION   = 8.0        
BETA_PENALTY    = 15.0       
L2_PENALTY      = 0.05       
TURNOVER_PENALTY= 0.02       
COV_WINDOW      = 126        
COV_SHRINKAGE   = 0.20       

# Sleeve-level position constraints
LONG_BUDGET         = 1.0    
SHORT_BUDGET        = 1.0    
MAX_POSITION_SIDE   = 0.20   
BETA_TOLERANCE      = 0.02  


# ─────────────────────────────────────────────
# NETWORK TOPOLOGY
# ─────────────────────────────────────────────

def _complete_adjacency(n: int) -> np.ndarray:
    return np.ones((n, n)) - np.eye(n)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _safe_zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.nanstd(values))
    if std < 1e-10:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.nanmean(values))) / std


def _build_alpha_vector(
    scores: pd.Series,
    tickers: list[str],
    signals: pd.Series,
) -> np.ndarray:
   
    alpha = np.zeros(len(tickers), dtype=float)

    long_idx  = [i for i, t in enumerate(tickers) if signals.get(t, 0) ==  1]
    short_idx = [i for i, t in enumerate(tickers) if signals.get(t, 0) == -1]

    if long_idx:
        long_scores = np.array([scores.get(tickers[i], 0.0) for i in long_idx], dtype=float)
        z = np.clip(_safe_zscore(long_scores), -3.0, 3.0)
        alpha[long_idx] = np.maximum(z, 0.0) + 1e-6

    if short_idx:
        short_scores = np.array([scores.get(tickers[i], 0.0) for i in short_idx], dtype=float)
        z = np.clip(_safe_zscore(-short_scores), -3.0, 3.0)
        alpha[short_idx] = -(np.maximum(z, 0.0) + 1e-6)

    return BASE_MU * alpha


def _estimate_covariance(
    returns_slice: pd.DataFrame | None,
    tickers: list[str],
) -> np.ndarray:
    
    n = len(tickers)

    if returns_slice is not None and not returns_slice.empty:
        data = returns_slice.reindex(columns=tickers).tail(COV_WINDOW).fillna(0.0)

        if len(data) >= 20:
            try:
                lw  = LedoitWolf().fit(data.values)
                cov = lw.covariance_
                # Blend with diagonal to add extra shrinkage
                diag = np.diag(np.diag(cov))
                cov  = (1.0 - COV_SHRINKAGE) * cov + COV_SHRINKAGE * diag
                cov += np.eye(n) * 1e-6
                return cov
            except Exception:
                pass

        # Thin-data fallback: diagonal from realized variance
        variances = data.var(axis=0).clip(lower=1e-6).values
        return np.diag(variances)

    # No data at all: unit diagonal
    return np.eye(n) * 0.04


def _cap_and_renormalize(weights: np.ndarray, cap: float) -> np.ndarray:
    if len(weights) == 0:
        return weights

    w   = np.maximum(weights.astype(float), 0.0)
    tot = float(w.sum())
    w   = w / tot if tot > 1e-12 else np.ones_like(w) / len(w)

    cap = max(float(cap), 1.0 / len(w) + 1e-6)
    for _ in range(30):
        over = w > cap
        if not np.any(over):
            break
        excess   = float((w[over] - cap).sum())
        w[over]  = cap
        under    = ~over
        if np.any(under):
            room     = np.maximum(cap - w[under], 0.0)
            room_sum = float(room.sum())
            if room_sum > 1e-12:
                w[under] += excess * (room / room_sum)
            else:
                w[under] += excess / int(np.sum(under))
        w  = np.maximum(w, 0.0)
        w /= max(float(w.sum()), 1e-12)

    return w / max(float(w.sum()), 1e-12)


def _project_weights(
    w: np.ndarray,
    signals_vec: np.ndarray,
    long_budget:       float = LONG_BUDGET,
    short_budget:      float = SHORT_BUDGET,
    max_position_side: float = MAX_POSITION_SIDE,
) -> np.ndarray:
   
    out        = np.array(w, dtype=float)
    long_mask  = signals_vec > 0
    short_mask = signals_vec < 0
    out[~(long_mask | short_mask)] = 0.0

    if np.any(long_mask):
        wl = _cap_and_renormalize(np.clip(out[long_mask], 0.0, None), max_position_side)
        out[long_mask] = wl * long_budget

    if np.any(short_mask):
        ws = _cap_and_renormalize(np.clip(-out[short_mask], 0.0, None), max_position_side)
        out[short_mask] = -ws * short_budget

    return out


# ─────────────────────────────────────────────
# BETA NEUTRALIZATION (sleeve-level)
# ─────────────────────────────────────────────

def _beta_neutralize_sleeve(
    w:           np.ndarray,
    betas_vec:   np.ndarray,
    signals_vec: np.ndarray,
    tolerance:   float = BETA_TOLERANCE,
) -> np.ndarray:
   
    out        = np.array(w, dtype=float)
    long_mask  = signals_vec > 0
    short_mask = signals_vec < 0

    if not np.any(long_mask) or not np.any(short_mask):
        return out

    long_budget  = float(np.clip(out[long_mask].sum(),   1e-9, None))
    short_budget = float(np.clip((-out[short_mask]).sum(), 1e-9, None))

    cap_long  = max(MAX_POSITION_SIDE, 1.0 / int(np.sum(long_mask))  + 1e-6)
    cap_short = max(MAX_POSITION_SIDE, 1.0 / int(np.sum(short_mask)) + 1e-6)

    for _ in range(30):
        net_beta = float(np.dot(out, betas_vec))
        if abs(net_beta) <= tolerance:
            break

        long_w  = np.clip(out[long_mask],  0.0, None)
        short_w = np.clip(-out[short_mask], 0.0, None)
        if long_w.sum() <= 1e-12 or short_w.sum() <= 1e-12:
            break

        long_w  /= long_w.sum()
        short_w /= short_w.sum()

        beta_l = np.clip(betas_vec[long_mask],  0.1, None)
        beta_s = np.clip(betas_vec[short_mask], 0.1, None)

        if net_beta > 0:
            target_long  = 1.0 / beta_l
            target_short = beta_s
        else:
            target_long  = beta_l
            target_short = 1.0 / beta_s

        target_long  /= max(float(target_long.sum()),  1e-12)
        target_short /= max(float(target_short.sum()), 1e-12)

        eta     = 0.20
        long_w  = (1.0 - eta) * long_w  + eta * target_long
        short_w = (1.0 - eta) * short_w + eta * target_short

        long_w  = _cap_and_renormalize(long_w,  cap_long)
        short_w = _cap_and_renormalize(short_w, cap_short)

        out[long_mask]  =  long_w  * long_budget
        out[short_mask] = -short_w * short_budget

    return out


# ─────────────────────────────────────────────
# SINGLE SLEEVE OPTIMIZER (CBO)
# ─────────────────────────────────────────────

def optimize_sleeve(
    tickers:          list[str],
    signals:          pd.Series,
    composite_scores: pd.Series,
    betas:            pd.Series | None = None,
    returns_slice:    pd.DataFrame | None = None,
    prev_weights:     pd.Series | None = None,
) -> pd.Series:
  
    active_tickers = [t for t in tickers if signals.get(t, 0) != 0]
    n = len(active_tickers)

    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        sign = float(np.sign(signals.get(active_tickers[0], 1)))
        return pd.Series({active_tickers[0]: sign})

    signals_vec = np.array(
        [float(signals.get(t, 0)) for t in active_tickers], dtype=float
    )
    scores    = composite_scores.reindex(active_tickers).fillna(0.0)
    alpha     = _build_alpha_vector(scores, active_tickers, signals)
    cov       = _estimate_covariance(returns_slice, active_tickers)
    betas_vec = (
        np.array([float(betas.get(t, 1.0)) for t in active_tickers], dtype=float)
        if betas is not None else np.ones(n, dtype=float)
    )
    prev_vec  = (
        np.array([float(prev_weights.get(t, 0.0)) for t in active_tickers], dtype=float)
        if prev_weights is not None else np.zeros(n, dtype=float)
    )

    w0         = np.zeros(n, dtype=float)
    long_mask  = signals_vec > 0
    short_mask = signals_vec < 0

    if np.any(long_mask):
        la  = np.clip(alpha[long_mask], 0.0, None)
        exp = np.exp(la / (np.std(la) + 1e-8))
        w0[long_mask] = exp / max(float(exp.sum()), 1e-12)

    if np.any(short_mask):
        sa  = np.clip(-alpha[short_mask], 0.0, None)
        exp = np.exp(sa / (np.std(sa) + 1e-8))
        w0[short_mask] = -(exp / max(float(exp.sum()), 1e-12))

    w0 = _project_weights(w0, signals_vec)

    x = np.tile(w0, (n, 1))

    A         = _complete_adjacency(n)
    D         = np.diag(A.sum(axis=1))
    L         = D - A
    mix_scale = EPSILON / max(float(np.max(np.diag(D))), 1.0)
    mix       = np.eye(n) - mix_scale * L   # consensus mixing matrix

    for _ in range(NSTEPS):
        x_mixed = mix @ x
        z       = x.mean(axis=0)          # population mean (shared signal)
        x_next  = np.zeros_like(x)

        for i in range(n):
            wi           = x[i]
            beta_exp     = float(np.dot(betas_vec, wi))

            grad = (
                -alpha
                + 2.0 * RISK_AVERSION  * (cov @ wi)
                + 2.0 * L2_PENALTY     * wi
                + 2.0 * BETA_PENALTY   * beta_exp * betas_vec
                +       TURNOVER_PENALTY * np.sign(wi - prev_vec)
            )

            wi_new = 0.75 * (wi - STEP_SIZE * grad) + 0.20 * x_mixed[i] + 0.05 * z
            wi_new = _project_weights(wi_new, signals_vec)
            x_next[i] = wi_new

        x = x_next

    w_star = x.mean(axis=0)
    w_star = _project_weights(w_star, signals_vec)

    if betas is not None:
        w_star = _beta_neutralize_sleeve(w_star, betas_vec, signals_vec)

    return pd.Series(
        {active_tickers[i]: float(w_star[i]) for i in range(n)}
    )


# ─────────────────────────────────────────────
# BETA POLISH
# ─────────────────────────────────────────────

def _portfolio_beta_polish(
    results_df: pd.DataFrame,
    betas:      pd.Series,
) -> pd.DataFrame:
    
    df       = results_df.copy()
    df['beta'] = df['ticker'].map(betas).fillna(1.0)

    for sector in df['sector'].unique():
        mask   = df['sector'] == sector
        sleeve = df[mask]
        active = sleeve[sleeve['signal'] != 0]
        if active.empty:
            continue

        tickers     = active['ticker'].tolist()
        signals_vec = active['signal'].to_numpy(dtype=float)
        w           = active['weight'].to_numpy(dtype=float)
        b           = active['beta'].to_numpy(dtype=float)

        w_adj = _beta_neutralize_sleeve(w, b, signals_vec)

        for i, tkr in enumerate(tickers):
            df.loc[
                (df['sector'] == sector) & (df['ticker'] == tkr), 'weight'
            ] = float(w_adj[i])

    return df


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def optimize_sleeves(
    screener_results: pd.DataFrame,
    betas:            pd.Series | None = None,
    returns:          pd.DataFrame | None = None,
    prev_weights:     pd.Series | None = None,
    verbose:          bool = True,
) -> pd.DataFrame:
    
    results             = screener_results.copy()
    results['weight']   = 0.0
    sectors             = results['sector'].unique()

    if verbose:
        print(f"\n[CBO] Optimizing {len(sectors)} sector sleeves...")
        print(
            f"      NSTEPS={NSTEPS}  STEP_SIZE={STEP_SIZE}  EPSILON={EPSILON}  "
            f"RISK_AVERSION={RISK_AVERSION}  BETA_PENALTY={BETA_PENALTY}  "
            f"TURNOVER_PENALTY={TURNOVER_PENALTY}"
        )

    for sector in sectors:
        mask   = results['sector'] == sector
        sleeve = results[mask].copy()
        active = sleeve[sleeve['signal'] != 0]

        if active.empty:
            continue

        tickers = active['ticker'].tolist()
        signals = active.set_index('ticker')['signal']
        scores  = active.set_index('ticker')['score']

        sector_returns = (
            returns.reindex(columns=tickers) if returns is not None else None
        )
        sector_prev = (
            prev_weights.reindex(tickers) if prev_weights is not None else None
        )

        optimized = optimize_sleeve(
            tickers,
            signals,
            scores,
            betas=betas,
            returns_slice=sector_returns,
            prev_weights=sector_prev,
        )

        for tkr, w in optimized.items():
            results.loc[
                (results['sector'] == sector) & (results['ticker'] == tkr), 'weight'
            ] = round(float(w), 6)

        if verbose:
            n_long  = int((active['signal'] ==  1).sum())
            n_short = int((active['signal'] == -1).sum())
            print(f"  [{sector}] {n_long} longs, {n_short} shorts → optimized")

    # Portfolio-level beta polish
    if betas is not None:
        results = _portfolio_beta_polish(results, betas)
        if verbose:
            print("\n[CBO] Portfolio-level beta polish applied.")

    if verbose:
        long_mask  = results['signal'] ==  1
        short_mask = results['signal'] == -1
        print(f"\n[CBO] Done.")
        print(f"  Total long weight  : {results.loc[long_mask,  'weight'].sum():.4f}")
        print(f"  Total short weight : {results.loc[short_mask, 'weight'].sum():.4f}")
        print(f"  Net exposure       : {results['weight'].sum():.4f}")

    return results