# 📊 Sharpe Ratio Optimization Tool

The **Sharpe Ratio Optimization Tool** is an interactive Streamlit web app that helps users construct optimal portfolios using Sharpe Ratio maximization. It combines **Monte Carlo simulation** with **constrained numerical optimization**, allowing you to account for **transaction costs** and **turnover penalties**, and benchmark your strategy against the **SPY ETF**.

## 🚀 Features

- 🎯 Optimize portfolios to **maximize Sharpe Ratio**
- 🎲 Simulate thousands of portfolios using **Monte Carlo**
- 🛡️ Apply **constraints** like max asset weight
- 🔁 Add **turnover penalty** and **transaction costs**
- 📈 Visualize **rolling Sharpe ratios** and **cumulative returns**
- 📉 Benchmark against **SPY**
- 💾 Download **CSV** of optimal allocations

## 📸 Preview

| Monte Carlo Simulation | Rolling Sharpe Ratio |
|------------------------|----------------------|
| ![MC](screenshots/monte_carlo_plot.png) | ![Rolling Sharpe](screenshots/rolling_sharpe.png) |

> _Optional: Create a `/screenshots` folder with `monte_carlo_plot.png` and `rolling_sharpe.png` screenshots to enable previews._

## 🧠 How It Works

1. Input tickers manually or select from a curated list of popular stocks.
2. Choose simulation and optimization parameters in the sidebar:
   - Start/End date
   - Risk-free rate
   - Max weight per asset
   - Number of portfolios to simulate
   - Turnover penalty
   - Transaction cost
3. The app:
   - Downloads historical data from **Yahoo Finance**
   - Simulates thousands of portfolios
   - Finds the best-performing portfolio by Sharpe Ratio
   - Runs a constrained optimization routine for even better results
   - Displays performance and download options

## 🛠️ Installation

### Requirements

- Python 3.8+
- streamlit
- yfinance
- pandas
- numpy
- matplotlib
- scipy

### Setup

```bash
git clone https://github.com/yourusername/sharpe-optimizer.git
cd sharpe-optimizer
pip install -r requirements.txt
streamlit run app.py
