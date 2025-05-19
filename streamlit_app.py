import streamlit as st
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import io

try:
    import riskfolio.Portfolio as pf
    riskfolio_available = True
except ImportError:
    riskfolio_available = False

st.set_page_config(page_title="Correlation Matrix App", layout="wide")
st.title("📈 Stock Correlation & Risk Dashboard")

# Sidebar controls
st.sidebar.header("Configuration")

# Available ticker options
ticker_options = ["ANET", "FN", "ALAB","AAPL", "TSLA", "MSFT", "AMZN", "NVDA"]
tickers = st.sidebar.multiselect("Select tickers", options=ticker_options, default=ticker_options)
start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-04-30"))
window = st.sidebar.slider("Rolling Correlation Window (days)", 20, 180, 60)

# Main analysis trigger
if st.sidebar.button("🔍 Run Analysis"):
    with st.spinner("Fetching and analyzing data..."):
        data = {}

        # Download each ticker independently to avoid MultiIndex
        for ticker in tickers:
            st.write(f"Downloading {ticker}...")
            stock = yf.download(ticker, start=start, end=end, group_by="column")

            if not stock.empty:
                if "Adj Close" in stock.columns:
                    data[ticker] = stock["Adj Close"].squeeze()
                elif "Close" in stock.columns:
                    st.warning(f"{ticker} missing 'Adj Close'. Using 'Close' instead.")
                    data[ticker] = stock["Close"].squeeze()
                else:
                    st.error(f"{ticker} has no valid price columns.")
            else:
                st.error(f"{ticker} returned no data.")

        if len(data) == 0:
            st.error("❌ No valid data downloaded.")
        else:
            df = pd.DataFrame(data).dropna()
            st.subheader("📊 Adjusted Close Prices")
            st.line_chart(df)

            # Return and correlation
            returns = df.pct_change().dropna()
            corr = returns.corr()

            st.subheader("📌 Correlation Matrix")
            st.dataframe(corr.round(3))

            st.subheader("🔴 Heatmap")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
            st.pyplot(fig)

            # CSV Export
            buffer = io.StringIO()
            corr.to_csv(buffer)
            buffer.seek(0)
            csv = buffer.getvalue().encode("utf-8")  # Convert to bytes
            st.download_button("⬇️ Download Correlation Matrix CSV", data=csv, file_name="correlation_matrix.csv", mime="text/csv")

            # Rolling correlation
            if len(tickers) >= 2:
                st.subheader(f"🔁 Rolling Correlation: {tickers[0]} vs {tickers[1]}")
                roll_corr = returns[tickers[0]].rolling(window).corr(returns[tickers[1]])
                st.line_chart(roll_corr.dropna())

            # Multi-Frequency Correlations
            st.subheader("🗓️ Multi-Frequency Correlations")

            # Daily
            st.markdown("### 📅 Daily Correlation")
            daily_returns = df.pct_change().dropna()
            daily_corr = daily_returns.corr()
            st.dataframe(daily_corr.round(3))
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(daily_corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
            st.pyplot(fig)

            # Monthly
            st.markdown("### 🗓️ Monthly Correlation")
            monthly_prices = df.resample('M').last()
            monthly_returns = monthly_prices.pct_change().dropna()
            monthly_corr = monthly_returns.corr()
            st.dataframe(monthly_corr.round(3))
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(monthly_corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
            st.pyplot(fig)

            # Quarterly
            st.markdown("### 📆 Quarterly Correlation")
            quarterly_prices = df.resample('Q').last()
            quarterly_returns = quarterly_prices.pct_change().dropna()
            quarterly_corr = quarterly_returns.corr()
            st.dataframe(quarterly_corr.round(3))
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(quarterly_corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
            st.pyplot(fig)

            # Yearly
            st.markdown("### 📈 Year-over-Year Correlation")
            yearly_prices = df.resample('Y').last()
            yoy_returns = yearly_prices.pct_change().dropna()
            yoy_corr = yoy_returns.corr()
            st.dataframe(yoy_corr.round(3))
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(yoy_corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
            st.pyplot(fig)
            
            # Risk metrics + optimization
            if riskfolio_available and len(tickers) > 1:
                st.subheader("📉 Risk Metrics (VaR, CVaR, Sharpe)")
                port = pf.Portfolio(returns=returns)
                port.assets_stats(method_mu='hist', method_cov='hist')
                risk = port.risk_measures(method='hist', rf=0)
                st.dataframe(risk[["VaR_0.05", "CVaR_0.05", "Sharpe"]].round(4))

                st.subheader("🧠 Portfolio Optimization (Max Sharpe)")
                w = port.optimization(model="Classic", rm="MV", obj="Sharpe", hist=True)
                st.dataframe(w.T.round(4))

                port_weights = w[w > 0].index.tolist()
                weighted_returns = returns[port_weights].mul(w.T[port_weights].values, axis=1).sum(axis=1)
                cumulative_returns = (1 + weighted_returns).cumprod()
                st.subheader("📈 Optimized Portfolio Cumulative Returns")
                st.line_chart(cumulative_returns)

                st.subheader("📉 Drawdown Chart")
                drawdown = (cumulative_returns - cumulative_returns.cummax()) / cumulative_returns.cummax()
                st.line_chart(drawdown)

                st.subheader("🔁 Monthly Rebalanced Portfolio Returns")
                rebalance_returns = weighted_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                st.line_chart((1 + rebalance_returns).cumprod())
            elif not riskfolio_available:
                st.warning("Install `riskfolio-lib` to enable risk metrics.")
    st.success("Analysis complete!")
else:
    st.info("👈 Select tickers and press 'Run Analysis' to begin.")

