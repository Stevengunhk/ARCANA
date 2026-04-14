# ◈ ARCANA

### Adaptive Regime-Constrained Arbitrage with Normalised Allocation

> A regime-aware statistical arbitrage framework applied to the S&P 100 universe (2013–2025).
> Net Sharpe 2.534 · Max Drawdown −5.33% · 13 consecutive positive years · Beta 0.009
<img width="2258" height="2450" alt="fig1_equity_curve" src="https://github.com/user-attachments/assets/f694fcc9-e7b1-4863-8df0-26616b636a82" />

---

## Overview

ARCANA combines three ideas that individually exist in the literature but have not previously been integrated as a complete system:

1. **Rolling cointegration on log prices** — 3-year training window, re-selected each year, sector-constrained
2. **HMM regime detection** — 3-state Gaussian Hidden Markov Model scales positions by market regime
3. **Dynamic volatility targeting** — daily vol scalar maintains constant 10% annualised portfolio risk

The result is a market-neutral equity strategy that traded the S&P 100 from January 2013 to December 2025, generating a net annualised return of 17.35% at 6.85% volatility after realistic transaction costs.

---

## Performance Summary

| Metric | Gross | Net (after TC + risk controls) |
|---|---|---|
| Annualised Return | 19.97% | **17.35%** |
| Annualised Volatility | 6.89% | **6.85%** |
| Sharpe Ratio | 2.898 | **2.534** |
| Maximum Drawdown | −4.78% | **−5.33%** |
| Calmar Ratio | 4.178 | **3.253** |
| Win Rate | 58.7% | **57.0%** |
| Cumulative Return | 1,193% | **821%** |
| Portfolio Beta | 0.002 | **0.009** |
| Transaction Cost Drag | — | 2.58% / yr |

### In-Sample vs Out-of-Sample

| Period | Sharpe | NW t-stat | p-value |
|---|---|---|---|
| In-Sample (2013–2020) | 2.722 | 7.124 | < 0.001 |
| Out-of-Sample (2021–2025) | **2.998** | 6.371 | < 0.001 |
| Alpha Decay | **29.1%** | — | — |

OOS Sharpe exceeds IS Sharpe. Alpha decay of 29.1% is within the conventional 30% robustness threshold.

### Statistical Significance (5/5 tests passed)

| Test | Statistic | p-value |
|---|---|---|
| Simple t-test | t = 9.153 | < 0.001 |
| Newey-West HAC (5 lags) | t = **9.454** | < 0.001 |
| Lo (2002) Sharpe test | z = 4.108 | < 0.001 |
| Bootstrap CI (10,000 draws) | SR ∈ [2.24, 3.39] | < 0.001 |
| Durbin-Watson autocorrelation | DW = **2.012** | — |

Durbin-Watson of 2.012 confirms returns are **not autocorrelated** — the Sharpe ratio is not inflated.

---

## Year-by-Year Net Returns

| Year | Net Return | Sharpe | Regime |
|---|---|---|---|
| 2013 | +35.7% | 3.41 | Bull |
| 2014 | +11.9% | 1.99 | Bull |
| 2015 | +16.2% | 2.40 | Neutral |
| 2016 | +13.1% | 2.47 | Bull |
| 2017 | +22.5% | 3.04 | Bull |
| 2018 | +15.6% | 2.62 | Bull |
| 2019 | +14.6% | 2.15 | Bull |
| 2020 | +12.3% | 1.68 | **Crisis** |
| 2021 | +21.6% | 3.39 | Bull |
| 2022 | +1.3% | 0.65 | **Crisis** |
| 2023 | +18.9% | 2.77 | Neutral |
| 2024 | +20.4% | 2.40 | Bull |
| 2025 | +21.6% | 3.56 | Neutral |

**All 13 years positive.** Including +12.3% in COVID-2020 and +1.3% in the 2022 rate shock.

---

## Architecture

```
Stage 1   Data Layer          101 tickers, S&P 100 + SPY, daily OHLCV
Stage 2   Feature Engineering FF3 residuals, returns, beta, vol
Stage 3   HMM Regime Engine   3-state Gaussian HMM on SPY returns + realised vol
Stage 4A  Stat Arb Engine     Rolling 3yr cointegration, sector-constrained, OU filtering
Stage 5   Signal Blending     Regime-gated Z-score signals
Stage 6   Portfolio            Pair-based backtest, dynamic vol targeting
Stage 7   Risk Management     TC model, circuit breakers, significance tests
Stage 8   Live Trading        IB paper trading via ib_insync (port 7497)
Stage 9   Monitoring          Daily HTML report, live P&L vs backtest
```

---

## Methodology

### 1. Universe & Sector Constraint

- **Universe:** S&P 100 constituent stocks (101 securities including SPY)
- **Sector constraint:** Only within-GICS-sector pairs are tested (8 sectors, ~675 candidate pairs per year)
- Eliminates spurious cointegrations between economically unrelated stocks

### 2. Rolling Walk-Forward Pair Selection

Each January, for trade year T:

```
Train window:  [T−3, T−1]   3 years of log price history
Test window:   [T, T]        trade the selected pairs in year T
```

**Selection filters applied to training window:**
- Engle-Granger cointegration test on log prices at p < 0.05
- OU half-life: 5–80 trading days
- OU R² ≥ 0.05
- Top 10 pairs by quality score = R² / half-life

### 3. Signal Generation

The spread is defined as:

```
S_t = log(P_{A,t}) − h · log(P_{B,t})
```

The rolling 60-day Z-score:

```
Z_t = (S_t − μ_60) / σ_60
```

**Entry:** |Z_t| > 2.0  
**Exit:** |Z_t| < 0.5  
**Stop-loss:** |Z_t| > 4.0

### 4. HMM Regime Filter

Three-state Gaussian HMM estimated on:
- SPY daily returns (weighted 2.5×)
- 20-day realised volatility of SPY

| State | Label | % of Days | Position Scale |
|---|---|---|---|
| 0 | Bull | 44.9% | 100% |
| 1 | Neutral | 36.2% | 70% |
| 2 | Crisis | 18.9% | **0% (flat)** |

No new positions are opened during Crisis regime.

### 5. Dynamic Volatility Targeting

```python
vol_scalar = min(VOL_TARGET / realised_vol_60d, 3.0)
weight      = BASE_WEIGHT × vol_scalar × regime_scale
```

Target: 10% annualised portfolio volatility.

---

## Pipeline File Structure

```
C:/Arbion Research/
├── Stage 1 data layer/Universe_stock data/     ← OHLCV CSVs per ticker
├── Stage 2 feature engineering/                ← features_returns.csv, features_market.csv
├── Stage 3 HMM regime engine/                  ← regime_labels_weekly.csv, regime_probs_smoothed.csv
├── Stage 4A stat arb engine/                   ← pairs_by_year_rolling.csv, zscore_signals_rolling.csv
│                                                  signals_gated_rolling.csv, hedge_ratios_rolling.csv
├── Stage 5 signal blending/                    ← blended_signal.csv
├── Stage 6 portfolio construction/             ← portfolio_returns.csv, portfolio_weights.csv
├── Stage 7 risk management/                    ← net_returns.csv, risk_report.csv
├── Stage 8 live trading/                       ← arcana_ib_engine.py, state.json
└── Stage 9 monitoring/                         ← daily_report_YYYY-MM-DD.html, live_pnl.csv
```

---

## Key Notebooks

| File | Stage | Description |
|---|---|---|
| `arcana_stage3_hmm.ipynb` | 3 | HMM regime detection |
| `arcana_stage4a_rolling.ipynb` | 4A | Rolling walk-forward pair selection |
| `arcana_stage6_portfolio_v3.ipynb` | 6 | Pair-based backtest |
| `arcana_stage7_risk.ipynb` | 7 | Risk management + significance tests |
| `arcana_ib_engine.py` | 8 | Live IB paper trading engine |
| `arcana_stage9_monitor.py` | 9 | Daily monitoring + HTML report |
| `arcana_backtest_video.py` | — | Equity curve animation |
| `arcana_3d_hmm_v3.py` | — | 3D HMM transition matrix video |

---

## Live Trading (Stage 8)

The IB engine connects to Interactive Brokers TWS (paper trading, port 7497) and runs daily at 09:35 ET:

```python
# Run daily at market open
python arcana_ib_engine.py
```

**What it does each day:**
1. Connects to TWS, reads account equity
2. Fetches live prices from IB for all active pairs
3. Detects current HMM regime
4. Computes rolling 60-day Z-scores
5. Opens / closes / stop-losses positions via VWAP execution
6. Saves state to `state.json`

```python
# Run daily at market close for monitoring report
python arcana_stage9_monitor.py
```

---

## Requirements

```
Python 3.10+
numpy
pandas
statsmodels
hmmlearn
scikit-learn
matplotlib
scipy
ib_insync          # for live trading (Stage 8)
```

```bash
pip install numpy pandas statsmodels hmmlearn scikit-learn matplotlib scipy ib_insync
```

---

## Research Paper

This strategy is documented in a working paper submitted to SSRN:

**Regime-Aware Statistical Arbitrage: Rolling Cointegration with Dynamic Volatility Targeting in the S&P 100**

> MSc Student, Imperial College London  
> April 2026

Key findings:
- Newey-West t-statistic: **9.454** (p < 0.001)
- Bootstrap 95% CI for Sharpe: **[2.24, 3.39]**
- Durbin-Watson: **2.012** (no autocorrelation)
- OOS Sharpe (2021–2025): **2.998** — exceeds in-sample
<img width="2637" height="1166" alt="fig2_regime_distributions" src="https://github.com/user-attachments/assets/8d7e08d6-67f8-4cc5-8149-ba0fdd0553ea" />
<img width="3012" height="1579" alt="fig3_pair_contributions" src="https://github.com/user-attachments/assets/1c5a3487-8311-46f1-8dbb-b607d33f1b24" />
<img width="2185" height="1809" alt="fig4_regime_probabilities" src="https://github.com/user-attachments/assets/3e4b1e8c-5c94-4916-bd83-b9023b608636" />

---

## Disclaimer

This repository is for research and educational purposes only. Past backtest performance does not guarantee future results. This is not investment advice. All live trading is conducted on Interactive Brokers paper trading accounts only.

---

*ARCANA — Adaptive Regime-Constrained Arbitrage with Normalised Allocation*  
*Imperial College London · April 2026*
