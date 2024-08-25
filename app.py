import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import datetime


# Function to load data
def load_data(stock_symbol, start_date, end_date):
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    return df

# Function to visualize data
def visualize_data(df, user_input):
    st.subheader('Closing Price vs Time Chart')
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    st.pyplot(fig1)

    st.subheader('Closing Price vs Time Chart With 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig2)

    st.subheader('Closing Price vs Time Chart With 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig3 = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(ma200)
    plt.plot(df.Close)
    st.pyplot(fig3)

    st.subheader('Prediction vs Original')
    # Load model
    model = load_model('keras_models.h5')
    
    # Preprocess data for prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])
    data_training_array = scaler.fit_transform(data_training)
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)
    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Make prediction
    y_predicted = model.predict(x_test)
    scaler = scaler.scale_
    scale_factor = 1 / scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Plot prediction vs original
    fig4 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig4)

# Function to compare stock prices for two dates
# Function to compare stock prices between two dates
def compare_dates(df, start_date, end_date):
    # Convert start_date and end_date to Pandas Timestamp
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    # Filter data for the selected date range
    df_filtered = df.loc[(df.index >= start_date) & (df.index <= end_date)]

    # Plot the comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_filtered.index, df_filtered['Close'], label='Closing Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Comparison of Stock Prices')
    ax.legend()
    st.pyplot(fig)



st.title("Stock Trend Prediction")

# Sidebar Navigation
nav_selection = st.sidebar.radio("Navigation", ("Graph", "Data", "Comparison", "Stock News", "List of Stock Tickers"))

if nav_selection == "Graph":
    st.header("Graph")
    user_input = st.text_input("Enter Stock Ticker", "")
    if st.button("Show Graphs"):
        if user_input == "":
            st.warning("Please enter a stock ticker.")
        else:
            start_date = '2010-01-01'  # Default start date
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')  # Default end date
            df = load_data(user_input, start_date, end_date)
            if df.empty:
                st.warning("No data available for the given stock ticker.")
            else:
                visualize_data(df, user_input)



elif nav_selection == "Data":
    st.header("Data")
    user_input = st.text_input("Enter Stock Ticker", "")
    if st.button("Show Data"):
        if user_input == "":
            st.warning("Please enter a stock ticker.")
        else:
            start_date = '2010-01-01'  # Default start date
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')  # Default end date
            df = load_data(user_input, start_date, end_date)
            if df.empty:
                st.warning("No data available for the given stock ticker.")
            else:
                st.subheader("Data from 2010-2019")
                st.write(df.describe())



elif nav_selection == "Comparison":
    st.header("Comparison")
    user_input = st.text_input("Enter Stock Ticker", "")
    start_date = st.date_input("Enter Start Date")
    end_date = st.date_input("Enter End Date")
    if st.button("Compare Dates"):
        if user_input == "":
            st.warning("Please enter a stock ticker.")
        else:
            df = load_data(user_input, start_date, end_date)
            if df.empty:
                st.warning("No data available for the given stock ticker and date range.")
            else:
                compare_dates(df, start_date, end_date)





elif nav_selection == "Stock News":
    st.header("Stock News")
    # You can add your code to fetch and display stock news here.
    # Function to fetch latest stock prices

    # Sample stock news headlines
    stock_news_headlines = [
        "Tech Giants Rally as Earnings Surpass Expectations",
        "Biotech Sector Sees Surge Following FDA Approval",
        "Energy Stocks Dip Amidst Global Supply Concerns",
        "Retail Sector Gains Momentum with Strong Q1 Sales",
        "Cryptocurrency Market Volatility Sparks Investor Interest",
        "Healthcare Stocks React to New Drug Breakthroughs",
        "Financial Sector Braces for Impact of Interest Rate Hikes",
        "Electric Vehicle Companies Expand Amid Growing Demand",
        "Defense Contractors Rally Amidst Geopolitical Tensions",
        "Consumer Goods Stocks Fluctuate in Response to Supply Chain Disruptions"
    ]

    # Display stock news headlines
    for headline in stock_news_headlines:
        st.write("- " + headline)

elif nav_selection == "List of Stock Tickers":
    st.header("List of Stock Tickers")
    # You can add a list of stock tickers with company names here.
    data = {
        "Company": ['Apple Inc.', 'Microsoft Corporation', 'Amazon.com Inc.', 'Alphabet Inc.', 'Facebook, Inc.',
                    'Tesla, Inc.', 'Berkshire Hathaway Inc.', 'Johnson & Johnson', 'JPMorgan Chase & Co.', 'Visa Inc.',
                    'Walmart Inc.', 'The Procter & Gamble Company', 'Coca-Cola Company', 'Pfizer Inc.', 'Exxon Mobil Corporation',
                    'Alibaba Group Holding Limited', 'The Walt Disney Company', 'Netflix, Inc.', 'Nike, Inc.', 'McDonald\'s Corporation',
                    'AT&T Inc.', 'Cisco Systems, Inc.', 'Intel Corporation', 'General Electric Company', 'Boeing Company',
                    'Adobe Inc.', 'NVIDIA Corporation', 'Mastercard Incorporated', 'Salesforce.com, Inc.',
                    'Goldman Sachs Group, Inc.', 'Caterpillar Inc.', 'PepsiCo, Inc.', '3M Company', 'Costco Wholesale Corporation',
                    'Home Depot, Inc.', 'Accenture plc', 'American Express Company', 'Wells Fargo & Company', 'Chevron Corporation',
                    'Merck & Co., Inc.', 'Bristol Myers Squibb Company', 'General Motors Company', 'Ford Motor Company', 'Honeywell International Inc.',
                    'UnitedHealth Group Incorporated', 'Verizon Communications Inc.', 'PayPal Holdings, Inc.', 'Starbucks Corporation', 'QUALCOMM Incorporated',
                    'Booking Holdings Inc.', 'Oracle Corporation', 'CVS Health Corporation', 'Visa Inc.', 'Netflix, Inc.', 'Square, Inc.', 'The Clorox Company',
                    'Target Corporation', 'Raytheon Technologies Corporation', 'Danaher Corporation', 'Mastercard Incorporated', 'Costco Wholesale Corporation',
                    'Activision Blizzard, Inc.', 'T-Mobile US, Inc.', 'NVIDIA Corporation', 'Biogen Inc.', 'Eli Lilly and Company', 'Amgen Inc.', 'Facebook, Inc.',
                    'Booking Holdings Inc.', 'Intuit Inc.', 'Capital One Financial Corporation', 'Deere & Company', 'Bristol Myers Squibb Company',
                    'Johnson & Johnson', 'American Airlines Group Inc.', '3M Company', 'Twitter, Inc.', 'Union Pacific Corporation',
                    'Micron Technology, Inc.', 'Salesforce.com, Inc.', 'United Parcel Service, Inc.', 'The Home Depot, Inc.', 'Morgan Stanley',
                    'Thermo Fisher Scientific Inc.', 'PayPal Holdings, Inc.', 'The Coca-Cola Company', 'Charter Communications, Inc.', 'Exxon Mobil Corporation',
                    'Broadcom Inc.', 'Netflix, Inc.', 'Snap Inc.', 'Ferrari N.V.'],

        "Symbol": ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA', 'BRK.A', 'JNJ', 'JPM', 'V',
                   'WMT', 'PG', 'KO', 'PFE', 'XOM', 'BABA', 'DIS', 'NFLX', 'NKE', 'MCD',
                   'T', 'CSCO', 'INTC', 'GE', 'BA', 'ADBE', 'NVDA', 'MA', 'CRM',
                   'GS', 'CAT', 'PEP', 'MMM', 'COST', 'HD', 'ACN', 'AXP', 'WFC', 'CVX',
                   'MRK', 'BMY', 'GM', 'F', 'HON', 'UNH', 'VZ', 'PYPL', 'SBUX', 'QCOM',
                   'BKNG', 'ORCL', 'CVS', 'V', 'NFLX', 'SQ', 'CLX', 'TGT', 'RTX', 'DHR',
                   'MA', 'COST', 'ATVI', 'TMUS', 'NVDA', 'BIIB', 'LLY', 'AMGN', 'FB',
                   'BKNG', 'INTU', 'COF', 'DE', 'BMY', 'JNJ', 'AAL', 'MMM', 'TWTR', 'UNP',
                   'MU', 'CRM', 'UPS', 'HD', 'MS', 'TMO', 'PYPL', 'KO', 'CHTR', 'XOM',
                   'AVGO', 'NFLX', 'SNAP', 'RACE']
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Add a search bar
    search_term = st.text_input("Search for a company:")

    # Filter the DataFrame based on the search term
    filtered_df = df[df["Company"].str.contains(search_term, case=False)]

    # Display the filtered DataFrame as a table
    st.write(filtered_df)

