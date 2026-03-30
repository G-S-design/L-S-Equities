import numpy as np
import pandas as pd
import scipy.optimize as sco
from sklearn.covariance import LedoitWolf

SECTOR_MAX = 0.35
RISKFREERATE = 0.04
ROLLING_DAYS = 252 * 3
HARD_FLOOR = 0.05  # Sector min is 5%


def clean_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    h = holdings.copy()
    h["sleeve"] = h["sleeve"].astype(str).str.strip()
    h["ticker"] = h["ticker"].astype(str).str.upper().str.strip()
    if "weight" in h.columns:
        h["weight"] = pd.to_numeric(h["weight"], errors="coerce")
    return h


def build_sector_returns(
    prices: pd.DataFrame,
    holdings: pd.DataFrame,
    method: str = "equal",
    rolling_days: int = ROLLING_DAYS,
) -> pd.DataFrame:
    """_summary_

    Args:
        prices (pd.DataFrame): _description_
        holdings (pd.DataFrame): _description_
        method (str, optional): _description_. Defaults to "equal".
        rolling_days (int, optional): _description_. Defaults to ROLLING_DAYS.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    h = clean_holdings(holdings)
    rets = prices.ffill().pct_change().dropna(how="all")

    if rolling_days and len(rets) > rolling_days:
        rets = rets.iloc[-rolling_days:]

    sector_rets = {}
    for sleeve, grp in h.groupby("sleeve"):
        tickers = [t for t in grp["ticker"].tolist() if t in rets.columns]
        if len(tickers) == 0:
            continue
        r = rets[tickers]
        if method == "equal" or "weight" not in grp.columns:
            sector_rets[sleeve] = r.mean(axis=1)
        elif method == "holdings":
            w = grp.set_index("ticker")["weight"].reindex(tickers).fillna(0.0)
            s = float(w.sum())
            if s <= 0:
                sector_rets[sleeve] = r.mean(axis=1)
            else:
                w = w / s
                sector_rets[sleeve] = r.mul(w.values, axis=1).sum(axis=1)
        else:
            raise ValueError("method must be 'equal' or 'holdings'")

    out = pd.DataFrame(sector_rets).dropna(how="all").ffill().dropna()
    if out.shape[1] < 2:
        raise ValueError("Requires at least 2 sleeves with valid return series to run MVO.")
    return out


def estimate_moments(
    sector_returns: pd.DataFrame, annualize: int = 252
) -> tuple[pd.Series, pd.DataFrame]:
    """_summary_

    Args:
        sector_returns (pd.DataFrame): Daily return series for each sector sleeve. Rows are trading days, columns are sector names. 
        annualize (int, optional): Number of trading days per year

    Returns
        tuple[pd.Series, pd.DataFrame]: _description_
    """
    mu_hist = sector_returns.mean() * annualize #Mean of daily return across trading days.
    lw = LedoitWolf().fit(sector_returns.values)
    cov = pd.DataFrame(
        lw.covariance_ * annualize,
        index=sector_returns.columns,
        columns=sector_returns.columns,
    )
    return mu_hist, cov


def black_litterman_returns(
    cov: pd.DataFrame,
    risk_aversion: float = 2.5,
) -> pd.Series:
    """Implied equilibrium returns from equal-weight prior. Pi = lambda * Sigma * w_eq."""
    n = len(cov)
    w_eq = np.ones(n) / n
    Pi = risk_aversion * cov.values @ w_eq
    return pd.Series(Pi, index=cov.index)


def risk_parity_floors(
    cov: pd.DataFrame,
    hard_floor: float = HARD_FLOOR,
    sleeve_floors: dict | None = None,
) -> pd.Series:
    """
    Per-sector lower bounds derived from volatility.

    Floor_i = hard_floor * (avg_vol / vol_i)

    Low-volatility sectors (Utilities, Consumer Defensive) receive a floor
    above hard_floor because the optimizer would otherwise underweight them
    relative to their diversification benefit.
    High-volatility sectors (Energy, Tech) are floored at hard_floor.

    sleeve_floors: optional dict of {sector_name: floor} overrides applied
    after risk-parity calculation, e.g. {"Energy": 0.07} to give Energy
    a higher floor without affecting any other sector.
    """
    vols = pd.Series(np.sqrt(np.diag(cov.values)), index=cov.index)
    avg_vol = vols.mean()
    raw_floors = hard_floor * (avg_vol / vols)

    # Clamp: floor can't be below hard_floor or above sector_max
    raw_floors = raw_floors.clip(lower=hard_floor, upper=SECTOR_MAX)

    # Apply named overrides — only the specified sectors are affected
    if sleeve_floors:
        for sector, floor_val in sleeve_floors.items():
            if sector in raw_floors.index:
                raw_floors[sector] = floor_val

    # Feasibility check: if floors sum > 1, scale them down proportionally
    floor_sum = raw_floors.sum()
    if floor_sum > 1.0:
        raw_floors = raw_floors * (1.0 / floor_sum) * 0.95  # 5% headroom

    return raw_floors


def mvo_max_sharpe_with_floors(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float,
    floors: pd.Series,
    sector_max: float = SECTOR_MAX,
) -> pd.Series:
    """
    Max-Sharpe MVO with per-sector lower bounds (risk-parity floors)
    and a uniform upper bound (sector_max).
    """
    mu = mu.copy()
    cov = cov.loc[mu.index, mu.index].copy()
    floors = floors.reindex(mu.index)
    n = len(mu)

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ mu.values)
        port_var = float(w @ cov.values @ w)
        port_vol = float(np.sqrt(max(port_var, 1e-12)))
        return -((port_ret - rf) / port_vol)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Per-sector bounds: (risk_parity_floor_i, sector_max)
    bnds = [(float(floors.iloc[i]), sector_max) for i in range(n)]

    # Warm start: begin at floors, allocate remaining budget proportionally to BL mu
    w0 = floors.values.copy()
    remaining = 1.0 - w0.sum()
    mu_above_floor = np.maximum(mu.values - rf, 1e-8)
    w0 += remaining * (mu_above_floor / mu_above_floor.sum())
    w0 = np.clip(w0, [b[0] for b in bnds], [b[1] for b in bnds])
    w0 = w0 / w0.sum()

    res = sco.minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bnds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(f"MVO failed: {res.message}")

    w = pd.Series(res.x, index=mu.index)
    w = w.clip(lower=floors)
    w = w / w.sum()
    return w


def write_sleeves_csv(
    weights: pd.Series, path: str = "sleeves.csv", rounding: int = 4
) -> pd.DataFrame:
    weights.index.name = "sleeve"
    out = weights.rename("sleeve_weight").reset_index()
    out["sleeve_weight"] = out["sleeve_weight"].round(rounding)
    out.to_csv(path, index=False)
    return out


def optimize_sector_weights_to_csv(
    prices: pd.DataFrame,
    holdings: pd.DataFrame,
    out_path: str = "sleeves.csv",
    rf: float = RISKFREERATE,
    method: str = "holdings",
    annualize: int = 252,
    rolling_days: int = ROLLING_DAYS,
    sector_max: float = SECTOR_MAX,
    risk_aversion: float = 2.5,
    hard_floor: float = HARD_FLOOR,
    sleeve_floors: dict | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    sector_returns = build_sector_returns(prices, holdings, method=method, rolling_days=rolling_days)
    _, cov = estimate_moments(sector_returns, annualize=annualize)
    mu_bl = black_litterman_returns(cov, risk_aversion=risk_aversion)
    floors = risk_parity_floors(cov, hard_floor=hard_floor, sleeve_floors=sleeve_floors)

    if verbose:
        vols = pd.Series(np.sqrt(np.diag(cov.values)), index=cov.index)
        print("\n=== Risk-Parity Floors Applied ===")
        floor_df = pd.DataFrame({"annualised_vol": vols.round(4), "floor": floors.round(4)})
        print(floor_df.sort_values("annualised_vol").to_string())

    w = mvo_max_sharpe_with_floors(mu_bl, cov, rf=rf, floors=floors, sector_max=sector_max)
    return write_sleeves_csv(w, out_path)


if __name__ == "__main__":
    import yfinance as yf

    dummy_holdings = pd.DataFrame(
        [
            ("Technology", "AAPL", 0.25), ("Technology", "MSFT", 0.20), ("Technology", "NVDA", 0.18),
            ("Technology", "AVGO", 0.15), ("Technology", "ORCL", 0.12), ("Technology", "ADBE", 0.10),

            ("Healthcare", "UNH", 0.22), ("Healthcare", "LLY", 0.20), ("Healthcare", "JNJ", 0.18),
            ("Healthcare", "MRK", 0.15), ("Healthcare", "ABBV", 0.13), ("Healthcare", "PFE", 0.12),

            ("Financial Services", "JPM", 0.22), ("Financial Services", "BAC", 0.18), ("Financial Services", "WFC", 0.16),
            ("Financial Services", "GS", 0.15), ("Financial Services", "MS", 0.15), ("Financial Services", "V", 0.14),

            ("Consumer Defensive", "PG", 0.22), ("Consumer Defensive", "KO", 0.18), ("Consumer Defensive", "PEP", 0.18),
            ("Consumer Defensive", "WMT", 0.16), ("Consumer Defensive", "COST", 0.14), ("Consumer Defensive", "PM", 0.12),

            ("Industrials", "CAT", 0.20), ("Industrials", "GE", 0.18), ("Industrials", "HON", 0.17),
            ("Industrials", "UPS", 0.15), ("Industrials", "BA", 0.15), ("Industrials", "LMT", 0.15),

            ("Communication Services", "GOOGL", 0.22), ("Communication Services", "META", 0.20),
            ("Communication Services", "NFLX", 0.16), ("Communication Services", "DIS", 0.15),
            ("Communication Services", "TMUS", 0.14), ("Communication Services", "CMCSA", 0.13),

            ("Energy", "XOM", 0.26), ("Energy", "CVX", 0.22), ("Energy", "COP", 0.16),
            ("Energy", "EOG", 0.14), ("Energy", "SLB", 0.12), ("Energy", "OXY", 0.10),

            ("Utilities", "NEE", 0.22), ("Utilities", "DUK", 0.18), ("Utilities", "SO", 0.17),
            ("Utilities", "AEP", 0.15), ("Utilities", "EXC", 0.14), ("Utilities", "SRE", 0.14),
        ],
        columns=["sleeve", "ticker", "weight"],
    )

    dummy_holdings["sleeve"] = dummy_holdings["sleeve"].astype(str).str.strip()
    dummy_holdings["ticker"] = dummy_holdings["ticker"].astype(str).str.upper().str.strip()
    dummy_holdings["weight"] = pd.to_numeric(dummy_holdings["weight"], errors="coerce")

    tickers = sorted(dummy_holdings["ticker"].unique().tolist())
    raw = yf.download(
        tickers,
        start="2018-01-01",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw.rename("Close").to_frame()

    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.80)).dropna(how="any")
    prices.columns = [c.upper() for c in prices.columns]
    dummy_holdings = dummy_holdings[dummy_holdings["ticker"].isin(prices.columns)].copy()

    out_df = optimize_sector_weights_to_csv(
        prices=prices,
        holdings=dummy_holdings,
        out_path="sleeves.csv",
        rf=0.04,
        method="holdings",
        rolling_days=252 * 3,
        sector_max=0.35,
        risk_aversion=2.5,
        hard_floor=0.05,
        sleeve_floors={"Energy": 0.07},  # Energy-specific override; all others use risk-parity from 5%
        verbose=True,
    )

    print("\n=== MVO Sleeve Weights — BL + Risk-Parity Floors ===")
    print(out_df.to_string(index=False))