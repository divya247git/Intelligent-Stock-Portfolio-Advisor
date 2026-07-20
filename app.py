"""
app.py
========
Streamlit front-end for the Intelligent Stock Portfolio Advisor.

Run with:  streamlit run app.py
"""
import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_advisor.config import CONFIG
from stock_advisor.data_fetcher import DataFetcher
from stock_advisor.recommendation_engine import RecommendationEngine
from stock_advisor.portfolio_optimizer import PortfolioOptimizer, compute_returns
from stock_advisor.paper_trading import PaperTradingAccount
from stock_advisor.reporting import build_monthly_report, report_to_markdown

st.set_page_config(page_title="Intelligent Stock Portfolio Advisor", layout="wide")

if "account" not in st.session_state:
    st.session_state.account = PaperTradingAccount()
if "fetcher" not in st.session_state:
    st.session_state.fetcher = DataFetcher()
if "engine" not in st.session_state:
    st.session_state.engine = RecommendationEngine(fetcher=st.session_state.fetcher)

st.title("📈 Intelligent Stock Portfolio Advisor")

tab_research, tab_portfolio, tab_paper, tab_reports = st.tabs(
    ["🔍 Stock Research", "⚖️ Portfolio Optimizer", "💵 Paper Trading", "📊 Monthly Report"]
)

# ---------------------------------------------------------------------- #
# TAB 1: Stock Research / Recommendation
# ---------------------------------------------------------------------- #
with tab_research:
    st.subheader("Buy / Hold / Sell Recommendation")
    col1, col2, col3 = st.columns([2, 1, 1])
    ticker = col1.text_input("Ticker", value="AAPL").upper().strip()
    include_lstm = col2.checkbox("Run LSTM forecast", value=True)
    include_social = col3.checkbox("Include social sentiment", value=False)

    if st.button("Analyze", type="primary"):
        with st.spinner(f"Analyzing {ticker} — fundamentals, technicals, news NLP"
                         + (", LSTM" if include_lstm else "") + "..."):
            try:
                rec = st.session_state.engine.analyze(
                    ticker, include_lstm=include_lstm, include_social=include_social
                )
                st.session_state[f"rec_{ticker}"] = rec
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                rec = None

        if rec:
            action_color = {"BUY": "green", "HOLD": "orange", "SELL": "red"}[rec.action]
            st.markdown(f"### Recommendation: :{action_color}[{rec.action}]  "
                        f"(composite score: {rec.composite_score:+.2f})")
            st.caption(rec.rationale)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Fundamental Score", f"{rec.fundamental_score:+.2f}")
            m2.metric("Technical Score", f"{rec.technical_score:+.2f}")
            m3.metric("News Sentiment", f"{rec.news_sentiment_score:+.2f}", f"n={rec.news_sample_size}")
            m4.metric("Social Sentiment", f"{rec.social_sentiment_score:+.2f}", f"n={rec.social_sample_size}")

            st.text(f"Fundamentals: {rec.fundamental_notes}")

            if rec.lstm_forecast:
                lf = rec.lstm_forecast
                st.markdown("#### LSTM Price Forecast")
                c1, c2, c3 = st.columns(3)
                c1.metric("Last Close", f"${lf.last_actual_price:,.2f}")
                c2.metric("Predicted Next Close", f"${lf.predicted_next_price:,.2f}",
                          f"{lf.predicted_return_pct:+.2f}%")
                c3.metric("Test RMSE", f"${lf.test_rmse:,.2f}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=lf.history_dates, y=lf.history_actual, name="Actual"))
                fig.add_trace(go.Scatter(x=lf.history_dates, y=lf.history_predicted, name="Predicted"))
                fig.update_layout(title="LSTM Backtest: Actual vs Predicted (test window)",
                                   xaxis_title="Date", yaxis_title="Price ($)")
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------- #
# TAB 2: Portfolio Optimizer (MPT)
# ---------------------------------------------------------------------- #
with tab_portfolio:
    st.subheader("Modern Portfolio Theory Allocation")
    tickers_input = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, GOOGL, AMZN, NVDA, JPM")
    risk_appetite = st.select_slider("Risk appetite", options=["conservative", "balanced", "aggressive"],
                                      value="balanced")
    max_weight = st.slider("Max weight per holding", 0.1, 1.0, 0.4, 0.05)

    if st.button("Optimize Portfolio", type="primary"):
        tick_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        with st.spinner("Pulling price history and solving for the efficient frontier..."):
            try:
                price_data = st.session_state.fetcher.get_multi_price_history(tick_list, days=730)
                returns = compute_returns(price_data)
                optimizer = PortfolioOptimizer()
                result = optimizer.recommend_by_risk_appetite(returns, risk_appetite, max_weight)
                frontier = optimizer.efficient_frontier(returns, n_points=30, max_weight=max_weight)
            except Exception as e:
                st.error(f"Optimization failed: {e}")
                result, frontier = None, []

        if result:
            st.markdown(f"### Suggested Allocation ({risk_appetite})")
            weights_df = pd.DataFrame(
                [{"Ticker": t, "Weight": f"{w*100:.1f}%"} for t, w in
                 sorted(result.weights.items(), key=lambda x: -x[1])]
            )
            c1, c2 = st.columns([1, 2])
            c1.dataframe(weights_df, hide_index=True, use_container_width=True)

            fig_pie = go.Figure(data=[go.Pie(labels=list(result.weights.keys()),
                                              values=list(result.weights.values()), hole=0.4)])
            c2.plotly_chart(fig_pie, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Expected Annual Return", f"{result.expected_annual_return*100:.2f}%")
            m2.metric("Annual Volatility", f"{result.annual_volatility*100:.2f}%")
            m3.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")

            if frontier:
                fig_frontier = go.Figure()
                fig_frontier.add_trace(go.Scatter(
                    x=[p.annual_volatility * 100 for p in frontier],
                    y=[p.expected_annual_return * 100 for p in frontier],
                    mode="markers+lines", name="Efficient Frontier",
                ))
                fig_frontier.add_trace(go.Scatter(
                    x=[result.annual_volatility * 100], y=[result.expected_annual_return * 100],
                    mode="markers", marker=dict(size=14, symbol="star", color="red"),
                    name="Your Portfolio",
                ))
                fig_frontier.update_layout(title="Efficient Frontier", xaxis_title="Volatility (%)",
                                            yaxis_title="Expected Return (%)")
                st.plotly_chart(fig_frontier, use_container_width=True)

# ---------------------------------------------------------------------- #
# TAB 3: Paper Trading
# ---------------------------------------------------------------------- #
with tab_paper:
    st.subheader("Paper Trading Simulator")
    account = st.session_state.account

    c1, c2, c3, c4 = st.columns(4)
    pt_ticker = c1.text_input("Ticker", value="AAPL", key="pt_ticker").upper().strip()
    pt_side = c2.selectbox("Side", ["BUY", "SELL"])
    pt_qty = c3.number_input("Quantity", min_value=0.0, value=10.0, step=1.0)
    pt_price = c4.number_input("Price ($)", min_value=0.0, value=0.0, step=0.01,
                                help="Leave 0 to use latest market price")

    if st.button("Submit Order", type="primary"):
        try:
            price = pt_price
            if price <= 0:
                df = st.session_state.fetcher.get_price_history(pt_ticker, days=5)
                price = float(df["close"].iloc[-1])
            trade = account.place_order(pt_ticker, pt_side, pt_qty, price)
            st.success(f"Executed {trade.side} {trade.quantity} {trade.ticker} @ ${trade.price:.2f}")
        except Exception as e:
            st.error(f"Order failed: {e}")

    st.markdown("### Account Snapshot")
    held_tickers = [t for t, p in account.positions.items() if p.quantity > 0]
    current_prices = {}
    for t in held_tickers:
        try:
            df = st.session_state.fetcher.get_price_history(t, days=5)
            current_prices[t] = float(df["close"].iloc[-1])
        except Exception:
            current_prices[t] = account.positions[t].avg_cost

    snap = account.snapshot(current_prices)
    m1, m2, m3 = st.columns(3)
    m1.metric("Portfolio Value", f"${snap['portfolio_value']:,.2f}", f"{snap['total_return_pct']:+.2f}%")
    m2.metric("Cash", f"${snap['cash']:,.2f}")
    m3.metric("Realized P&L", f"${snap['realized_pnl']:,.2f}")

    if snap["positions"]:
        st.dataframe(pd.DataFrame(snap["positions"]).T, use_container_width=True)

    if account.trade_log:
        st.markdown("### Trade Blotter")
        st.dataframe(pd.DataFrame([t.__dict__ for t in account.trade_log]), use_container_width=True)

# ---------------------------------------------------------------------- #
# TAB 4: Monthly Report
# ---------------------------------------------------------------------- #
with tab_reports:
    st.subheader("Monthly Performance & Attribution Report")
    today = dt.date.today()
    default_start = (today.replace(day=1) - dt.timedelta(days=1)).replace(day=1)
    start_date = st.date_input("Period start", value=default_start)
    end_date = st.date_input("Period end", value=today)

    if st.button("Generate Report", type="primary"):
        account = st.session_state.account
        held_tickers = list(account.positions.keys())
        if not held_tickers:
            st.warning("No positions in the paper trading account yet — place some trades first.")
        else:
            with st.spinner("Pulling price history and computing attribution..."):
                price_history = st.session_state.fetcher.get_multi_price_history(held_tickers, days=730)
                report = build_monthly_report(
                    account, price_history, str(start_date), str(end_date)
                )
            st.markdown(report_to_markdown(report))
