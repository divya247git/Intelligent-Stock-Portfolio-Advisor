📈 Intelligent Stock Portfolio Advisor

Overview

The Intelligent Stock Portfolio Advisor is an AI-powered investment platform that helps users make informed stock market decisions by combining financial analysis, machine learning, natural language processing, and portfolio optimization techniques. The system analyzes stock fundamentals, technical indicators, market sentiment from news and social media, and historical price trends to generate actionable Buy, Hold, or Sell recommendations.

The platform also provides stock price forecasting using LSTM neural networks, optimizes portfolio allocation based on investor risk tolerance using Modern Portfolio Theory (MPT), and includes a paper trading simulator for risk-free strategy testing.


---

Features

🔍 Fundamental Analysis

Analyze company financial statements and key ratios.

Evaluate earnings, revenue growth, P/E ratio, debt-to-equity ratio, and profitability metrics.

Determine the financial health of stocks.


📊 Technical Analysis

Calculate popular technical indicators:

Moving Average (MA)

Relative Strength Index (RSI)

MACD

Bollinger Bands

Volume Trends


Identify potential entry and exit points.


📰 Sentiment Analysis

Extract financial news and social media discussions.

Use Transformer-based NLP models (BERT/FinBERT) for sentiment classification.

Categorize sentiment as Positive, Neutral, or Negative.


🤖 Stock Price Prediction

Train LSTM (Long Short-Term Memory) models on historical stock data.

Forecast future stock prices and trends.

Support short-term and medium-term predictions.


💡 Recommendation Engine

Combine:

Fundamental Analysis

Technical Analysis

Sentiment Analysis

Price Prediction


Generate Buy, Hold, or Sell recommendations.


💼 Portfolio Optimization

Implement Modern Portfolio Theory (MPT).

Calculate risk-return tradeoffs.

Generate optimal asset allocations based on:

Conservative Risk

Moderate Risk

Aggressive Risk



🎮 Paper Trading Simulator

Simulate stock trading without real money.

Test and evaluate investment strategies.

Track virtual portfolio performance.


📈 Performance Reporting

Generate monthly portfolio reports.

Analyze portfolio returns and risk metrics.

Provide attribution analysis showing which investment decisions improved or reduced returns.



---

Technology Stack

Frontend

React.js

HTML5

CSS3

Bootstrap


Backend

Python

Flask


Machine Learning & AI

TensorFlow

Keras

Scikit-Learn

Hugging Face Transformers


Data Analysis

Pandas

NumPy

Matplotlib

Plotly


Data Sources

Yahoo Finance API

Alpha Vantage API

Financial News APIs

Social Media APIs



---

Project Structure

Intelligent-Stock-Portfolio-Advisor/
│
├── app.py
├── config.py
├── data_fetcher.py
├── fundamental_analysis.py
├── technical_indicators.py
├── sentiment_analysis.py
├── lstm_predictor.py
├── recommendation_engine.py
├── portfolio_optimizer.py
├── paper_trading.py
├── reporting.py
├── requirements.txt
└── README.md


---

Workflow

1. Collect stock market, news, and social media data.


2. Perform data preprocessing and feature engineering.


3. Analyze stock fundamentals and technical indicators.


4. Conduct sentiment analysis using NLP models.


5. Predict future prices using LSTM networks.


6. Generate Buy, Hold, or Sell recommendations.


7. Optimize portfolio allocation using MPT.


8. Simulate trades through paper trading.


9. Produce monthly performance and attribution reports.




---

Installation

Clone the Repository

git clone https://github.com/Intelligent-Stock-Portfolio-Advisor.git
cd Intelligent-Stock-Portfolio-Advisor

Install Dependencies

pip install -r requirements.txt

Run the Application

python app.py


---

Output 

<img width="1532" height="765" alt="image" src="https://github.com/user-attachments/assets/46e975d6-08dd-4fcf-9580-f2b32e01fa03" />
<img width="1650" height="765" alt="image" src="https://github.com/user-attachments/assets/388c4daa-5b02-4292-b34b-ecafc2e45f80" />
<img width="1650" height="627" alt="image" src="https://github.com/user-attachments/assets/4c5d2668-f012-4175-9671-2583e47be7a4" />
<img width="1650" height="775" alt="image" src="https://github.com/user-attachments/assets/5100df38-0196-4b8a-b132-8018f156827b" />
<img width="1650" height="521" alt="image" src="https://github.com/user-attachments/assets/c35be0ef-5848-45a9-9ae6-8d801847ba1d" />
<img width="1650" height="652" alt="image" src="https://github.com/user-attachments/assets/77a1ccbf-a54b-4f82-9f41-fbd0d6aa567b" />


Future Enhancements

Real-time stock alerts

Cryptocurrency portfolio support

Reinforcement learning-based trading strategies

AI-powered financial assistant chatbot

Multi-market support (US, India, Europe)

ESG investment recommendations



---

Target Users

Retail Investors

Financial Advisors

Wealth Managers

Investment Clubs

Finance Students and Researchers



---

Expected Outcomes

Improved investment decision-making

AI-driven portfolio recommendations

Enhanced risk management

Better portfolio diversification

Transparent and explainable investment insights



---

License

This project is licensed under the MIT License.


---

Author

Divyanshi Yadav
B.Tech CSE (AI & Deep Learning)
Mody University of Science and Technology

⭐ If you find this project useful, consider giving it a star on GitHub!
