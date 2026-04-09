# Quant Trading System

Systematic trading engine with realistic execution and multi-strategy portfolio allocation.

## Overview
This project implements a cost-aware backtesting system for equity trading strategies, focusing on realistic execution constraints and risk management.

## Key Features
- Momentum and regime-based strategies
- T+1 execution to avoid lookahead bias
- Transaction cost and slippage modeling
- Dynamic portfolio allocation (70/30 split)
- Risk metrics: Sharpe ratio, drawdown, CAGR

## Results
- CAGR: 22.37%
- Max Drawdown: -19.41%
- Sharpe Ratio: 1.86
- Annualized Volatility:: 11.41

## Strategy Logic
The system combines momentum and trend-following signals with regime filters to reduce risk during bearish markets.

## Files
- `strategy.py` → main trading engine
- `quant_summary.pdf` → strategy explanation and results

## How to Run
pip install -r requirements.txt  
python strategy.py
