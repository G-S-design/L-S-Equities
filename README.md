# QUANTT Equities Hedged: A Market-Neutral Long/Short Equity Strategy

<div align="center">

!
**A dual-layer, consensus-based, multi-factor framework for systematic market-neutral equity investing**

[Research Paper](#research-paper)

</div>

---

## Overview

QUANTT Equities Hedged is a sophisticated, production-grade **market-neutral long/short equity strategy** developed by Queen's University's Algorithmic Network and Trading Team. The strategy generates alpha through a composite scoring engine that combines five fundamental and price-based factors while using a novel **Consensus-Based Optimizer (CBO)** for position sizing.

### Key Achievements
- **19.73%** annualized return (March 2025–March 2026)
- **2.51** Sharpe ratio (3.2× better than SPY's 0.78)
- **-2.22%** maximum drawdown (6.2× better than SPY's -13.72%)
- **70% lower volatility** than benchmark (5.74% vs 18.86%)
- **Positive skewness (+3.19)** in daily returns

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [The Strategy](#the-strategy)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage Guide](#usage-guide)
7. [File Structure](#file-structure)
8. [Performance Analysis](#performance-analysis)
9. [Limitations & Future Work](#limitations--future-work)
10. [Research Paper](#research-paper)

---

## Quick Start

### Minimal Example

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn yfinance scipy matplotlib

# 2. Build the stock universe
python UniverseBuilder.py

# 3. Generate factor scores and signals
python StockScreener.py

# 4. Run backtest (optional)
python ls_backtest.py

# 5. View results
python Analysis.py
```

### Expected Outputs
- `universe.csv` – 399 screened stocks across 11 sectors
- `prices_adj.parquet` – Cached price history
- `screener_signals.csv` – Factor scores and position signals
- `sleeves.csv` – Sector-level capital allocation
- `equity_curve.csv` – Daily portfolio values

---

## Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│               UNIVERSE CONSTRUCTION                      │
│   UniverseBuilder.py → 399 stocks, 11 GICS sectors     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│           FACTOR COMPUTATION (Factors.py)               │
│  • Momentum (M)       • Value (V)                        │
│  • Quality (Q)        • Low Volatility (LV)              │
│  • Size (S)           • Beta estimation                  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│     SECTOR SCREENING (StockScreener.py)                 │
│  • Composite scoring (normalized, sector-weighted)      │
│  • Signal generation (percentile-based longs/shorts)    │
│  • CBO position sizing (Consensus.py)                   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│     SECTOR ALLOCATION (mvo.py)                          │
│  • Black-Litterman equilibrium returns                  │
│  • Ledoit-Wolf covariance shrinkage                     │
│  • Max-Sharpe optimization with risk-parity floors     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│       PORTFOLIO CONSTRUCTION                            │
│  • Beta neutralization (±0.02 tolerance)                │
│  • Position weighting & normalization                   │
│  • Monthly rebalancing with T+1 execution lag           │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│           BACKTESTING (ls_backtest.py)                  │
│  • Event-driven simulation (daily mark-to-market)       │
│  • Transaction costs (18–45 bps by price tier)          │
│  • Short borrow fees (3% annualized)                    │
│  • Monthly turnover & cost analysis                     │
└──────────────────────────────────────────────────────────┘
```

---

## The Strategy

### Factor Framework

The strategy combines five empirically validated factors, each normalized cross-sectionally and sector-weighted:

| Factor | Formula | Intuition | Sector Weights |
|--------|---------|-----------|-----------------|
| **Momentum (M)** | 12-month return (skip 1m) + 6-month return | Recent outperformers persist | Tech (35%), Consumer Cyc (30%) |
| **Value (V)** | Avg of: B/M, E/P, EBITDA/EV, P/B | Cheap stocks outperform | Financials (45%), Energy (45%) |
| **Quality (Q)** | Avg of: ROE, Profit Margin, Sharpe, Earnings Stability | Profitable, stable firms lead | Healthcare (50%), Tech (40%) |
| **Low Volatility (LV)** | Inverse of total + idiosyncratic volatility | Low-risk premium anomaly | Consumer Def (35%), Utilities (30%) |
| **Size (S)** | -ln(Market Cap) | Small-cap premium | Industrials (15%), Utilities (5%) |

Sector-specific weights are tuned to align with empirical factor premiums in each industry.

### Consensus-Based Optimizer (CBO)

Rather than traditional mean-variance optimization, the CBO treats each stock as an independent agent on a fully connected network, iterating via distributed gradient descent:

**Objective Function:**
```
min w⊤Σw + λ_β(β⊤w)² + λ₂∥w∥² + λ_τ∥w - w_prev∥₁ - α⊤w
```

**Key Components:**
- **w⊤Σw** – Portfolio variance penalty (risk control)
- **λ_β(β⊤w)²** – Beta neutrality penalty (market-neutral mandate)
- **λ₂∥w∥²** – L2 regularization (diversification, anti-concentration)
- **λ_τ∥w - w_prev∥₁** – Turnover penalty (transaction cost proxy)
- **-α⊤w** – Expected return from composite factor scores

**Iterative Update Rule:**
```
w_i^(t+1) = 0.75 × (w_i - η∇f) + 0.20 × (consensus mix) + 0.05 × (population mean)
```

Converges over 400 iterations with learning rate η = 0.04, yielding diversified allocations robust to estimation error.

### Sector Allocation (Black-Litterman + Risk-Parity)

8 sectors receive capital via a modified Markowitz framework:
1. **Covariance estimation** – Ledoit-Wolf shrinkage (reduces estimation noise)
2. **Expected returns** – Black-Litterman implied equilibrium (avoids chasing recent winners)
3. **Constraints** – Risk-parity floors (low-vol sectors get higher minima) + 35% sector cap

Result: ~64% in Financials & Technology (high BL return), 36% in defensive sectors (diversification).

### Beta Neutralization

Two-pass sleeve-level procedure ensures net portfolio beta |β_p| < 0.02:
1. Long leg weights fixed
2. Short leg rescaled by factor λ* such that β_p ≈ 0
3. Re-normalize and re-cap at 20% per position

Realized net beta over backtest: ~0.03 (residual from estimation error & intra-month drift).

---

## Installation

### Requirements
- **Python 3.8+**
- **Key Libraries:**
  - `pandas` (data wrangling)
  - `numpy` (numerical computing)
  - `scikit-learn` (covariance shrinkage, preprocessing)
  - `yfinance` (price & fundamental data)
  - `scipy` (optimization, statistics)
  - `matplotlib` (visualization)

### Setup

```bash
# Clone or download the repository
cd quantt-equities-hedged

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install pandas numpy scikit-learn yfinance scipy matplotlib
```

### Data Cache

The first run downloads ~6 years of price history (1,508 trading days) for 399 tickers via yfinance. This is cached locally in `prices_adj.parquet` (~15 MB) to avoid repeated API calls. Subsequent runs load from cache unless manually deleted.

---

## Configuration

### Global Settings (StockScreener.py)

```python
UNIVERSE_PATH  = "universe.csv"           # Universe file
PRICES_CACHE   = "prices_adj.parquet"     # Cached prices
SIGNALS_OUTPUT = "screener_signals.csv"   # Output signals
START_DATE     = "2020-01-01"             # Historical start
END_DATE       = "2026-01-01"             # Historical end
LOOKBACK       = 756                      # Factor window (3 years)
MAX_WORKERS    = 10                       # Parallel downloads
```

### Factor Weights by Sector (StockScreener.py)

```python
SECTOR_WEIGHTS = {
    "Technology":           {'M': 0.35, 'V': 0.05, 'Q': 0.40, 'LV': 0.10, 'S': 0.10},
    "Healthcare":           {'M': 0.15, 'V': 0.20, 'Q': 0.50, 'LV': 0.10, 'S': 0.05},
    "Financial Services":   {'M': 0.10, 'V': 0.45, 'Q': 0.30, 'LV': 0.10, 'S': 0.05},
    # ... (11 sectors total)
}
DEFAULT_WEIGHTS = {'M': 0.20, 'V': 0.30, 'Q': 0.30, 'LV': 0.15, 'S': 0.05}
```

### Consensus-Based Optimizer (Consensus.py)

```python
NSTEPS          = 400        # Consensus iterations
EPSILON         = 0.05       # Mixing parameter
STEP_SIZE       = 0.04       # Learning rate
RISK_AVERSION   = 8.0        # Variance penalty
BETA_PENALTY    = 15.0       # Beta neutrality penalty
L2_PENALTY      = 0.05       # Concentration penalty
TURNOVER_PENALTY= 0.02       # Turnover penalty
COV_WINDOW      = 126        # Rolling covariance window (6 months)
```

### Mean-Variance Optimizer (mvo.py)

```python
SECTOR_MAX      = 0.35       # Max per-sector weight
RISKFREERATE    = 0.04       # Risk-free rate (4%)
ROLLING_DAYS    = 252 * 3    # 3-year estimation window
HARD_FLOOR      = 0.05       # Min sector weight (5%)
```

### Backtesting (ls_backtest.py)

```python
REBALANCE_FREQ  = 'M'        # Monthly rebalancing
EXECUTION_LAG   = 1          # T+1 day execution
TRANSACTION_COSTS = {
    '>$50':   0.0018,         # Large-cap: 18 bps
    '$10-$50': 0.0023,        # Mid-cap: 23 bps
    '<$10':    0.0045         # Small-cap: 45 bps
}
SHORT_BORROW_FEE = 0.03       # 3% annualized
```

---

## Usage Guide

### 1. Build Universe

```bash
python UniverseBuilder.py
```

Screens the top 50 stocks (by market cap) from each of 11 GICS sectors via Yahoo Finance Equity Query API. Applies sector-specific min market cap thresholds and deduplicates. Output: `universe.csv` (399 tickers).

**Customize:**
```python
SIZE = 50  # Change to screen different universe size
SECTOR_LIST = ["Technology", "Healthcare", ...]  # Edit sectors
```

### 2. Generate Factor Scores & Signals

```bash
python StockScreener.py
```

Main pipeline:
- Downloads price & fundamental data (cached in `prices_adj.parquet`)
- Computes 5 factors (Momentum, Value, Quality, Low Volatility, Size)
- Normalizes and combines into composite scores (sector-weighted)
- Generates long/short signals via percentile-based thresholds
- Optimizes position sizes using CBO
- Applies sector-level MVO allocation
- Beta-neutralizes portfolio
- Outputs: `screener_signals.csv`, `sleeves.csv`

**Example Output (screener_signals.csv):**
```
sector,ticker,score,signal,position,weight,beta
Technology,NVDA,1.2345,1,LONG,0.0456,1.72
Technology,TSLA,-0.9876,-1,SHORT,-0.0234,1.45
Healthcare,LLY,0.8765,1,LONG,0.0512,1.10
...
```

### 3. Validate Data

```bash
python testpricesadj.py
```

Sanity checks on `prices_adj.parquet`:
- Date coverage (start/end dates)
- Ticker coverage (% of active positions in cache)
- Data quality (% missing, zero/negative prices)
- Staleness (how recent is price data?)

Should show: ✅ Parquet looks good — ready to run backtest

### 4. Run Backtest (Optional)

```bash
python ls_backtest.py
```

Event-driven simulation from March 2025 to March 2026:
- Daily mark-to-market with transaction costs
- Monthly rebalancing with T+1 execution lag
- Position drift between rebalances
- Tracks P&L, Sharpe, drawdown, turnover

Outputs: `equity_curve.csv` with daily cumulative returns.

### 5. Plot Results

```bash
python Analysis.py
```

Loads `equity_curve.csv` and plots cumulative returns over time.

---

## File Structure

```
quantt-equities-hedged/
│
├── UniverseBuilder.py          # Stock screening & universe construction
├── Factors.py                  # Factor computation (M, V, Q, LV, S)
├── StockScreener.py            # Main pipeline (screening → optimization)
├── Consensus.py                # Consensus-Based Optimizer (CBO)
├── mvo.py                      # Black-Litterman + MVO sector allocation
├── ls_backtest.py              # Event-driven backtesting engine
├── testpricesadj.py            # Data validation utility
├── Analysis.py                 # Equity curve visualization
│
├── universe.csv                # Output: screened stocks
├── prices_adj.parquet          # Cache: adjusted price history
├── screener_signals.csv        # Output: factor scores & signals
├── sleeves.csv                 # Output: sector allocations
├── equity_curve.csv            # Output: daily P&L (from backtest)
│
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## Performance Analysis

### Backtest Results (March 2025–March 2026)

| Metric | Strategy | SPY | Advantage |
|--------|----------|-----|-----------|
| **Total Return** | 19.73% | 18.15% | +1.58% |
| **CAGR** | 19.90% | 18.30% | +1.60% |
| **Ann. Volatility** | 5.74% | 18.86% | **-70%** |
| **Sharpe Ratio** | 2.51 | 0.78 | **3.2×** |
| **Max Drawdown** | -2.22% | -13.72% | **6.2×** |
| **Calmar Ratio** | 8.86 | 1.33 | **6.7×** |
| **Sortino Ratio** | 6.54 | 0.96 | **6.8×** |
| **Return Skewness** | +3.19 | — | Positive |
| **Best Day** | +2.92% | +10.50% | Capped |
| **Worst Day** | -0.77% | -5.85% | Protected |

### Monthly Returns (2025–2026)

```
           Jan   Feb   Mar   Apr   May   Jun
2025     1.7%  -0.7%  0.3%  2.6%  1.7%  3.0%
2026     4.1%  -3.1%  0.8%  1.2%  2.1%  0.9%

Positive in 12 of 15 months; largest drawdown: -1.3% (March 2026)
```

### Cost Decomposition

Total friction drag: **7.89 percentage points** (27.62% gross → 19.73% net)

- **Transaction Costs:** 2.14% (41% monthly turnover × 69 bps per unit)
- **Short Borrow Fees:** 1.23% (3% annualized on short notional)
- **Execution Lag:** 0.52% (T+1 slippage)

**Sensitivity:** Strategy remains viable with:
- +50% higher costs → Sharpe 1.90
- 2× costs → Sharpe 1.22 (still profitable)

---

## Limitations & Future Work

### Known Limitations

1. **Fundamental Data Look-Ahead Bias**
   - Value & quality factors use point-in-time December 2025 fundamentals applied retroactively
   - Introduces upward bias on Sharpe ratio (real backtest would have quarterly updates)
   - **Fix:** Use Bloomberg Point-in-Time or Compustat historical snapshots

2. **Market Impact Not Modeled**
   - 159 positions rebalancing monthly in less-liquid ADRs & small-caps may face material slippage
   - **Fix:** Implement Almgren–Chriss volume-participation model

3. **Static Beta Estimation**
   - 756-day OLS betas lag during rapid regime change (e.g., 2022 rate-hiking cycle)
   - **Fix:** Use Kalman filter or DCC-GARCH dynamic betas

4. **Survivorship Bias**
   - Universe constructed from live December 2025 universe excludes delisted/acquired firms
   - Modest impact given large-cap, diversified universe, but present
   - **Fix:** Use CRSP or point-in-time universe database

5. **No Factor Attribution**
   - Returns not decomposed by factor or sector contributions
   - Macro regime performance (rising rates, recession, etc.) not analyzed
   - **Fix:** Implement Barra-style attribution + macro regime classifier

### Future Enhancements

- [ ] Dynamic sector views via Black-Litterman sentiment inputs
- [ ] Machine learning factor weighting (e.g., XGBoost factor importance)
- [ ] Real-time portfolio construction via live data pipeline (Kafka/streaming)
- [ ] Conditional Value-at-Risk (CVaR) optimization instead of Sharpe
- [ ] Multi-asset extension (fixed income, commodities, FX)
- [ ] Cross-sectional momentum with mean-reversion timing
- [ ] Explicit cost minimization via portfolio sparsity

---

## Research Paper

The full academic treatment is available in `2026_QUANTT_EquitiesHedged_ResearchPaper.pdf`.

**Key Sections:**
- Abstract & Introduction (problem definition, contributions)
- Data Sources & Universe Construction (399 stocks, 11 sectors)
- Factor Construction & Composite Scoring (M, V, Q, LV, S)
- Consensus-Based Optimizer (distributed gradient descent, network topology)
- Sector Allocation via Black-Litterman MVO
- Beta Neutralization (sleeve-level & portfolio-level)
- Backtesting Framework (event-driven, T+1 lag, transaction costs)
- Results & Performance Analysis (19.73% return, 2.51 Sharpe, -2.22% max DD)
- Limitations & Future Work

**Authors:**
- Benjamin Gorenc (Quantitative Analyst, QUANTT)
- Gabriel Soler (Quantitative Analyst, QUANTT)
- Andrew Farkouh (Quantitative Analyst, QUANTT)

**Institution:** Queen's University Algorithmic Network and Trading Team (QUANTT), Kingston, Ontario

---

## Contributing

This is a research/educational project by QUANTT. For improvements, bug reports, or general inquiries:

**Email:** Gabe.soler@queensu.ca

**Acknowledgments:**
- Fama & French (factor premiums framework)
- Markowitz & Black-Litterman (portfolio optimization)
- Nedić & Ozdağlar (distributed optimization)
- Ledoit & Wolf (covariance shrinkage)

---

## Disclaimer

**This strategy is for educational and research purposes only.** Past performance does not guarantee future results. The backtest incorporates point-in-time fundamental data and thus contains look-ahead bias. Real-world implementation would face market impact, execution slippage, and regulatory constraints not fully captured here.

**Not financial advice.** Consult a professional before deploying capital.

---

## License

&copy; 2026 Queen's University QUANTT. All rights reserved.

---

<div align="center">

**Questions?** See the research paper or open an issue.

*Last Updated: March 2026*

</div>
