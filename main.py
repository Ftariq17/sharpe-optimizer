import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime as dt
from scipy.optimize import minimize

# Page Setup
st.set_page_config(page_title="Portfolio Optimization", layout="wide")
st.title("Sharpe Ratio Optimization Tool")

# Sidebar Inputs
st.sidebar.header("Settings")

# Text input for custom tickers
ticker_input = st.sidebar.text_input("Enter Tickers (comma-separated)", value="AAPL, MSFT, GOOGL")

# Popular highly traded stocks list
popular_tickers = [
  "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
  "JPM", "MS", "GS", "BRK-B", "V", "BAC",
  "XOM", "UNH", "PFE", "WMT", "KO", "PEP",
  "COST", "NFLX", "CRM", "AVGO"
]

# Optional multiselect
popular_picks = st.sidebar.multiselect("Or choose from popular stocks:", popular_tickers)

# Combine and deduplicate tickers
tickers = list(set([t.strip().upper() for t in ticker_input.split(",") if t.strip()] + popular_picks))

start_date = st.date_input("Start Date", dt.date(2020, 1, 1))
end_date = st.date_input("End Date", dt.date(2025, 1, 1))
risk_free_rate = st.number_input("Risk-Free Rate", value=0.0395)
num_portfolios = st.sidebar.slider("Monte Carlo Simulations", min_value=10000, max_value=50000, value=20000, step=1000)
max_weight = st.sidebar.slider("Max Weight Per Asset", min_value=0.1, max_value=1.0, value=0.4, step=0.05)
turnover_penalty = st.sidebar.slider("Turnover Penalty", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
st.sidebar.caption("📉 Higher values discourage large portfolio changes.")
transaction_cost = st.sidebar.number_input("Transaction Cost (%)", min_value=0.0, max_value=1.0, value=0.05) / 100

if len(tickers) < 2:
    st.warning("Please select at least 2 tickers")
    st.stop()

@st.cache_data
def load_data(tickers, start, end):
    raw_data = yf.download(tickers + ["SPY"], start=start, end=end)["Close"]
    raw_data = raw_data.dropna()

    valid_tickers = []
    for ticker in tickers:
        if ticker in raw_data.columns and raw_data[ticker].notna().sum() >= 100:
            valid_tickers.append(ticker)
        else:
            st.warning(f"⚠️ '{ticker}' returned insufficient or missing data and was excluded.")

    price_df = raw_data[valid_tickers + ["SPY"]] if "SPY" in raw_data.columns else raw_data[valid_tickers]

    return price_df, valid_tickers

price_df, tickers = load_data(tickers, start_date, end_date)
log_returns = np.log(price_df / price_df.shift(1)).dropna()
annual_mean_returns = log_returns[tickers].mean() * 252
cov_matrix = log_returns[tickers].cov() * 252
num_assets = len(tickers)

# Monte Carlo Simulation
results = []
attempts = 0
while len(results) < num_portfolios and attempts < num_portfolios * 10:
    weights = np.random.dirichlet(np.ones(num_assets))
    if np.any(weights > max_weight):
        attempts += 1
        continue
    ret = np.dot(weights, annual_mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - risk_free_rate) / vol
    results.append([ret, vol, sharpe] + list(weights))
    attempts += 1

df_mc = pd.DataFrame(results, columns=["Return", "Volatility", "Sharpe"] + tickers)

if df_mc.empty:
    st.warning("⚠️ No Monte Carlo portfolios met the max weight constraint. Try increasing the max weight.")
    st.stop()

best_mc = df_mc.loc[df_mc["Sharpe"].idxmax()]

# Optimizer
def penalized_negative_sharpe(weights, mean_returns, cov_matrix, rf, prev_weights, turnover_penalty, transaction_cost):
    weights = np.array(weights)
    port_return = np.dot(weights, mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    turnover = np.sum(np.abs(weights - prev_weights))
    cost = turnover * transaction_cost
    effective_return = port_return - cost
    penalty = turnover_penalty * turnover
    return -((effective_return - rf) / port_volatility - penalty)

prev_weights = np.array([1.0 / num_assets] * num_assets)
constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
bounds = tuple((0, max_weight) for _ in range(num_assets))
initial_guess = best_mc[tickers].values

opt_result = minimize(
    penalized_negative_sharpe,
    initial_guess,
    args=(annual_mean_returns, cov_matrix, risk_free_rate, prev_weights, turnover_penalty, transaction_cost),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

opt_weights = opt_result.x
opt_return = np.dot(opt_weights, annual_mean_returns)
opt_volatility = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
turnover = np.sum(np.abs(opt_weights - prev_weights))
cost = turnover * transaction_cost
effective_return = opt_return - cost
opt_sharpe = (effective_return - risk_free_rate) / opt_volatility

rounded_weights = np.round(opt_weights * 100, 2)
rounded_weights[rounded_weights < 0.01] = 0
weights_df = pd.DataFrame({"Ticker": tickers, "Allocation (%)": rounded_weights})

# Output Results
st.subheader("🔧 Optimized Portfolio (Max Sharpe via Optimization)")
for i, t in enumerate(tickers):
    st.write(f"**{t}:** {opt_weights[i]:.2%}")
st.write(f"**Expected Return:** {opt_return:.2%}")
st.write(f"**Volatility:** {opt_volatility:.2%}")
st.write(f"**Sharpe Ratio:** {opt_sharpe:.4f}")
st.write(f"**Turnover:** {turnover:.2%}")
st.write(f"**Transaction Cost Incurred:** {cost:.2%}")

st.subheader("🎲 Best Portfolio (from Monte Carlo Simulation)")
for t in tickers:
    st.write(f"**{t}:** {best_mc[t]:.2%}")
st.write(f"**Expected Return:** {best_mc['Return']:.2%}")
st.write(f"**Volatility:** {best_mc['Volatility']:.2%}")
st.write(f"**Sharpe Ratio:** {best_mc['Sharpe']:.4f}")

# Rolling Sharpe Ratio
window = 60
rolling_returns = (log_returns[tickers] @ opt_weights).dropna()
rolling_vol = rolling_returns.rolling(window).std() * np.sqrt(252)
rolling_mean = rolling_returns.rolling(window).mean() * 252
rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_vol

rolling_mc = (log_returns[tickers] @ best_mc[tickers].values).dropna()
rolling_vol_mc = rolling_mc.rolling(window).std() * np.sqrt(252)
rolling_mean_mc = rolling_mc.rolling(window).mean() * 252
rolling_sharpe_mc = (rolling_mean_mc - risk_free_rate) / rolling_vol_mc

# Plotly Monte Carlo
fig_mc = px.scatter(df_mc, x="Volatility", y="Return", color="Sharpe", color_continuous_scale="viridis", width=700, height=450)
fig_mc.add_trace(go.Scatter(x=[opt_volatility], y=[opt_return], mode='markers', marker=dict(color='red', size=14, symbol='star'), name='Optimized', showlegend=True))
fig_mc.add_trace(go.Scatter(x=[best_mc["Volatility"]], y=[best_mc["Return"]], mode='markers', marker=dict(color='blue', size=12, symbol='x'), name='Best MC', showlegend=True))
fig_mc.update_layout(legend=dict(x=0.75, y=0.95))
st.plotly_chart(fig_mc)

# Plotly Rolling Sharpe
fig_sharpe = go.Figure()
fig_sharpe.add_trace(go.Scatter(y=rolling_sharpe, name="Optimized Portfolio", line=dict(color="green", width=2)))
fig_sharpe.add_trace(go.Scatter(y=rolling_sharpe_mc, name="Best MC Portfolio", line=dict(color="#39FF14", dash="dash", width=1)))
fig_sharpe.update_layout(title="Rolling Sharpe Ratio (60-day window)", yaxis_title="Sharpe Ratio", width=700, height=350)
st.plotly_chart(fig_sharpe)

# Plotly Backtest
cumulative_opt = (1 + rolling_returns).cumprod()
rolling_returns_mc = (log_returns[tickers] @ best_mc[tickers].values).dropna()
cumulative_mc = (1 + rolling_returns_mc).cumprod()
cumulative_spy = (1 + log_returns["SPY"]).cumprod()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=cumulative_opt, name="Optimized Portfolio", line=dict(color="green", width=2)))
fig2.add_trace(go.Scatter(y=cumulative_mc, name="Best MC Portfolio", line=dict(color="#39FF14", dash="dash", width=1)))
fig2.add_trace(go.Scatter(y=cumulative_spy, name="SPY Benchmark", line=dict(color="blue", width=1)))
fig2.update_layout(title="Backtested Cumulative Returns", yaxis_title="Portfolio Value", width=700, height=350)
st.plotly_chart(fig2)

# CSV Download
st.subheader("⬇️ Download Allocation")
st.dataframe(weights_df)
st.download_button("Download as CSV", weights_df.to_csv(index=False), file_name="optimized_portfolio.csv")




