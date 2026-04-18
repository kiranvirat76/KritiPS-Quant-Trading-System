# Quant Trading System

Systematic long-only trading strategy with realistic execution modeling,regime filtering and evaluated based on out-of-sample validation.

---

## Overview

This project implements a quantitative trading system focused on long-only equity strategies.
The objective is to capture medium- to long-term momentum while controlling downside risk using regime filters and volatility-aware position sizing.
The system is designed with real-world trading constraints, including execution delay (T+1), transaction costs, and market regime changes.
Tested on Indian equity markets (2010–2025) and the strategy performance was further
evaluated on unseen data (2021-2025) in an external competition setting.

---

## Key Features

- Long-only momentum strategy
- T+1 execution to eliminate lookahead bias
- Transaction cost and slippage modeling (0.268% per trade)
- Delisting and missing data handling
- Lower circuit / execution constraints modeled
- Regime-based exposure control using market breadth indicators
- Volatility-adjusted position sizing
- Walk-forward and out-of-sample validation
- Monte Carlo simulation for robustness analysis

---

## Strategy Design

The system combines multiple components:

### 1. Momentum Layer
- Kalman-filter smoothed price for noise reduction.
- 1-month, 6-month, and 12-month returns
- Cross-sectional ranking across stocks

### 2. Trend Filtering
- Moving averages to align with broader market direction

### 3. Regime Detection
- Market breadth indicators
- Bear and crash regime identification

### 4. Risk Management
- Volatility-based position sizing  
- Trailing stop-loss  
- Maximum position constraints  
- Portfolio diversification  

---

## Data

- NSE equities universe  
- OHLC price data  
- Multi-year historical period  
- Daily frequency with realistic execution assumptions  

---

## In-Sample Results 

| Metric                 |In-Sample|
|------------------------|---------|
| CAGR                   | 22.37%  |
| Sharpe Ratio           | 1.86    |
| Max Drawdown           | -19.41% |
| Annualized Volatility  | 11.41%  |
| Win Rate               | 55.66%  |
|Kriti Rank              | 2nd     |

---

## Realism Constraints

- T+1 execution (signals executed next day)  
- Transaction cost: 0.268% per trade  
- No lookahead bias  
- Delisting and missing data handled  
- Lower circuit execution constraints modeled  

---

## Why This Strategy Works

- Momentum persists due to behavioral biases and slow information diffusion  
- Trend filters avoid weak market regimes  
- Regime-based exposure reduces drawdowns in bearish conditions

---

## Limitations

- Does not model full market impact  
- Assumes sufficient liquidity  
- Performance may degrade in highly efficient markets  

---

## Repository Structure

## Files
- `strategy.py` → main trading engine
- `quant_summary.pdf` → strategy explanation and results

## How to Run
pip install -r requirements.txt  
python strategy.py

---

## License

MIT License
