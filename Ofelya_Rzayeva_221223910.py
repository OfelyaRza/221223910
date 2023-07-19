# Course - 	Financial Data Analytics in Python
# Casestudy
"""
@author: Ofelya Rzayeva 221223910
"""

import time
import requests
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np
from pandas import json_normalize
from datetime import datetime, timezone, timedelta
import matplotlib.dates as mdates


# API credentials
ClientID = "5_I8ZjVe"
ClientSecret = "4U0aLbG9o5OMCCKoVdKwJRVKzswNRGIVItfoDXYibT0"

#This function get_market(instrument) is defined to retrieve market data for a given instrument.
#It makes a GET request to the Deribit API endpoint with the specified instrument name. If the response
#status code is 200 (indicating a successful request), the market data is extracted and stored in
#a DataFrame. The market price is then extracted from the DataFrame and rounded to the nearest tenth.
#Finally, the current market price is printed and returned from the function.

# Helper function to get market data
def get_market(instrument):
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_instrument"
    params = {
        "instrument_name": instrument
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        print('Download market data was successful')
    else:
        print(f"Failed to retrieve data: {response.status_code}")

    market_data = pd.DataFrame(data["result"])
    market_price = int(market_data['mark_price'].values[0].round(-1))
    
    print(f"Current market price: {market_price}")
    
    return market_price

#This function sell(params) is defined to place a sell order. It makes a GET request to the specified
#URL endpoint (https://test.deribit.com/api/v2/private/sell) with the given parameters (params)
#and API authentication using the ClientID and ClientSecret.

# Helper function to place a sell order
def sell(params):
    url = "https://test.deribit.com/api/v2/private/sell"
    response = requests.get(url, params=params, auth=(ClientID, ClientSecret))
    
    if response.status_code == 200:
        data = response.json()
        print(f"Trade was successful: {data['result']}")
    else:
        print(f"Failed to trade: {response.status_code}")

# Calculate breakout prices
instrument_name = "BTC-PERPETUAL" #is set to "BTC-PERPETUAL" representing the instrument or asset.
threshold_percentage = 2.0 #is set to 2.0, indicating a 2% threshold for the breakout.

# Get the current market price
market_price = get_market(instrument_name) #The code then calls the function to retrieve the current market price of the instrument.

# Calculate breakout threshold
threshold = market_price * threshold_percentage / 100 #Next, the breakout threshold is calculated by multiplying the market price by the threshold percentage divided by 100.

# Determine breakout prices
breakout_high = market_price + threshold
breakout_low = market_price - threshold

#Finally, the breakout high and breakout low prices are determined by adding and 
#subtracting the threshold from the market price, respectively. The calculated breakout 
#prices are then printed.
print("Breakout prices:")
print(f"Breakout High: {breakout_high}")
print(f"Breakout Low: {breakout_low}")


# Define the parameters for the sell order
params = {
    "amount": market_price, # is set to the market price, which represents the amount to be sold.
    "instrument_name": instrument_name, #is set to the instrument name ("BTC-PERPETUAL") for which the sell order is placed.
    "type": "market" #is set to "market" indicating a market order.
}

# Place the sell order
sell(params) #function is then called, passing the params dictionary as an argument.This function executes
             #a sell order using the specified parameters and provides feedback on the success or failure of the trade.
             
class DataHandler:
    def __init__(self, db_name):
        self.engine = create_engine(f'sqlite:///{db_name}')  # Create a SQLite database engine

    def download(self, instrument_name):
        url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
        params = {
            "instrument_name": instrument_name,
            "end_timestamp": int(time.time() * 1000),  # In Epoch milliseconds
            "start_timestamp": int((time.time() - 1e6) * 1000),  # In Epoch milliseconds
            "resolution": "60"  # Minute data
        }
        response = requests.get(url, params=params)  # Send a GET request to the API
        data = response.json()  # Extract the JSON response

        print(data)  # Print the response data to inspect its structure

        if 'error' in data:  # Check if the response contains an error
            print(f"Error: {data['error']['message']}")
            return

        if 'result' in data and instrument_name in data['result']:
            ohlc_data = data['result'][instrument_name]  # Extract the OHLC data
            ohlc = pd.DataFrame(ohlc_data)  # Create a DataFrame from the OHLC data
            ohlc['timestamp'] = pd.to_datetime(ohlc['ticks'], unit='ms')  # Convert the timestamp to datetime format
            ohlc['instrument_name'] = instrument_name  # Add instrument name column
            ohlc['resolution'] = 60  # Add resolution column
            ohlc.to_sql('ohlc', self.engine, if_exists='replace')  # Store the data in the 'ohlc' table of the database
            print(f"Downloaded data for {instrument_name}")
        else:
            print(f"No data found for {instrument_name}")

    def select(self, query):
        return pd.read_sql(query, self.engine)  # Execute a SQL query and return the results as a DataFrame

    def plot(self, query):
        df = self.select(query)  # Retrieve data based on the query
        df.set_index('timestamp', inplace=True)  # Set 'timestamp' as the index
        df.plot()  # Plot the data
        plt.title(f'{query}')  # Set the title of the plot
        plt.xlabel('Date')  # Set the label for the x-axis
        plt.show()  # Display the plot

dh = DataHandler('07_datam.db')  # Create an instance of DataHandler with the specified database name

data = dh.select('SELECT * FROM ohlc')  # Retrieve the data from the 'ohlc' table in the database

# Calculate the simple moving average
lookback_period = 10
data['SMA'] = data['close'].rolling(window=lookback_period).mean()
# In this line, I calculate the simple moving average (SMA) using a specified lookback period.
# The SMA is calculated by taking the mean of the 'close' prices over the given window.

# Generate the signal for the strategy
data['Signal'] = np.where(data['close'] >= data['SMA'], 1, -1)
# Here, I generate the trading signal for the strategy based on a condition.
# If the 'close' price is greater than or equal to the SMA, the signal is set to 1 (indicating a buy signal).
# Otherwise, the signal is set to -1 (indicating a sell signal).

# Backtest the strategy
data['Returns'] = np.log(data['close'] / data['close'].shift(1))
data['Strategy Returns'] = data['Returns'] * data['Signal'].shift(1).fillna(0)
# In this part, I backtest the strategy by calculating the returns of each trade.
# I calculate the logarithmic returns based on the 'close' prices.
# The strategy returns are then computed by multiplying the returns with the previous trading signal.
# Missing values in the signal are filled with 0 to handle the initial case.

# Calculate cumulative returns
data['Cumulative Returns'] = data['Strategy Returns'].cumsum()
# Here, I calculate the cumulative returns of the strategy by taking the cumulative sum of the strategy returns.

# Plot the cumulative returns
data['timestamp'] = pd.to_datetime(data['timestamp']) # Convert the 'timestamp' column to datetime format.
data.plot(x='timestamp', y='Cumulative Returns', figsize=(10, 6), legend=True)
plt.title('Cumulative Returns - Single Moving Average Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.show()
# This code block plots the cumulative returns of the strategy over time.
# It creates a line plot with the timestamp on the x-axis and cumulative returns on the y-axis.
# The plot visualizes the performance of the strategy.

# Calculate strategy performance metrics
total_trades = data['Signal'].value_counts().sum()
# Count the total number of trades by summing the occurrences of each signal (buy/sell) in the 'Signal' column.

winning_trades = data.loc[data['Strategy Returns'] > 0, 'Strategy Returns'].count()
# Count the number of winning trades by selecting the trades with positive strategy returns and counting them.

win_rate = winning_trades / total_trades
# Calculate the win rate by dividing the number of winning trades by the total number of trades.

annualized_return = np.exp(data['Returns'].sum() * 252) - 1
# Calculate the annualized return by summing the log returns over the entire period and converting it to an annual return.

annualized_volatility = data['Returns'].std() * np.sqrt(252)
# Calculate the annualized volatility by multiplying the standard deviation of log returns by the square root of the number of trading days.

sharpe_ratio = annualized_return / annualized_volatility
# Calculate the Sharpe ratio by dividing the annualized return by the annualized volatility.

# Print performance metrics
print('Strategy Performance Metrics:')
print(f'Total Trades: {total_trades}')
print(f'Win Rate: {win_rate:.2%}')
print(f'Annualized Return: {annualized_return:.2%}')
print(f'Annualized Volatility: {annualized_volatility:.2%}')
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
# Print the calculated performance metrics of the strategy.
# The metrics include the total number of trades, win rate, annualized return, annualized volatility, and Sharpe ratio.


# Calculate the Bollinger Bands Backtest
window = 20
data['SMA'] = data['close'].rolling(window).mean()  # Calculate the simple moving average
data['STD'] = data['close'].rolling(window).std()  # Calculate the standard deviation
data['Upper Band'] = data['SMA'] + 2 * data['STD']  # Calculate the upper band
data['Lower Band'] = data['SMA'] - 2 * data['STD']  # Calculate the lower band

# Generate the signal for the strategy
data['Signal'] = np.where(data['close'] > data['Upper Band'], -1, np.where(data['close'] < data['Lower Band'], 1, 0))
# Generate the trading signal based on the price crossing above the upper band (-1) or below the lower band (1).

# Calculate the strategy returns
data['Strategy Returns'] = data['Returns'] * data['Signal'].shift(1).fillna(0)
# Calculate the strategy returns by multiplying the daily returns with the trading signal from the previous day.

# Calculate cumulative returns
data['Cumulative Returns'] = (data['Strategy Returns'] + 1).cumprod() - 1
# Calculate the cumulative returns by taking the cumulative product of the strategy returns.

# Plot the cumulative returns
data.plot(x='timestamp', y='Cumulative Returns', figsize=(10, 6))
plt.title('Bollinger Bands Breakout Backtest - Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.show()
# Plot the cumulative returns over time to visualize the performance of the Bollinger Bands breakout strategy.


# Calculate RSI
window = 14
delta = data['close'].diff()  # Calculate the price change between each period
gain = delta.where(delta > 0, 0)  # Calculate the gain by setting negative values to 0
loss = -delta.where(delta < 0, 0)  # Calculate the loss by setting positive values to 0
average_gain = gain.rolling(window).mean()  # Calculate the average gain over the specified window
average_loss = loss.rolling(window).mean()  # Calculate the average loss over the specified window
relative_strength = average_gain / average_loss  # Calculate the relative strength
rsi = 100 - (100 / (1 + relative_strength))  # Calculate the RSI

# Generate the signal for the strategy
threshold = 30
data['Signal'] = np.where(rsi < threshold, 1, np.where(rsi > 70, -1, 0))
# Generate the trading signal based on the RSI crossing below the lower threshold (1) or above the upper threshold (-1).

# Calculate the strategy returns
data['Strategy Returns'] = data['Returns'] * data['Signal'].shift(1).fillna(0)
# Calculate the strategy returns by multiplying the daily returns with the trading signal from the previous day.

# Calculate cumulative returns
data['Cumulative Returns'] = (data['Strategy Returns'] + 1).cumprod() - 1
# Calculate the cumulative returns by taking the cumulative product of the strategy returns.

# Plot the cumulative returns for RSI Strategy
data.plot(x='timestamp', y='Cumulative Returns', figsize=(10, 6))
plt.title('RSI Strategy Backtest - Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.show()
# Plot the cumulative returns over time to visualize the performance of the RSI strategy.


# Calculate Breakout Startegy
short_window = 12
long_window = 26
signal_period = 9

#In the context of MACD (Moving Average Convergence Divergence), the values short_window, long_window, and
#signal_period represent the time periods used to calculate the various moving averages and signal line.
#short_window: It refers to the number of periods used to calculate the shorter-term exponential moving average (EMA) in MACD. In this case, it is set to 12.
#long_window: It represents the number of periods used to calculate the longer-term exponential moving average (EMA) in MACD. In this case, it is set to 26.
#signal_period: It indicates the number of periods used to calculate the signal line, which is a smoothed moving average of the MACD line. In this case, it is set to 9.

ema_short = data['close'].ewm(span=short_window, adjust=False).mean()  # Calculate the exponential moving average for the short window
ema_long = data['close'].ewm(span=long_window, adjust=False).mean()  # Calculate the exponential moving average for the long window
macd = ema_short - ema_long  # Calculate the MACD line
signal_line = macd.ewm(span=signal_period, adjust=False).mean()  # Calculate the signal line

# Generate the signal for the strategy
data['Signal'] = np.where(macd > signal_line, 1, np.where(macd < signal_line, -1, 0))
# Generate the trading signal based on the MACD line crossing above the signal line (1) or below the signal line (-1).

# Calculate the strategy returns
data['Strategy Returns'] = data['Returns'] * data['Signal'].shift(1).fillna(0)
# Calculate the strategy returns by multiplying the daily returns with the trading signal from the previous day.

# Calculate cumulative returns
data['Cumulative Returns'] = (data['Strategy Returns'] + 1).cumprod() - 1
# Calculate the cumulative returns by taking the cumulative product of the strategy returns.

# Plot the cumulative returns for MACD Strategy
data.plot(x='timestamp', y='Cumulative Returns', figsize=(10, 6))
plt.title('MACD Strategy Backtest - Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.show()
# Plot the cumulative returns over time to visualize the performance of the MACD strategy.

#The main aim of running this code is to visualize the trade positions from the transaction log data.
#The code processes the transaction log data and creates a scatter plot to represent the trade
#positions of different types: open sell, close buy, and short.

# Read the CSV file
df = pd.read_csv('transaction_log.csv')

# Convert 'Date' column to pandas Timestamp
df['Date'] = pd.to_datetime(df['Date'])

# Set the desired start date
start_date = pd.Timestamp(2023, 7, 9, 10, 11)

# Filter the dataframe for the desired start date and actions
df_filtered = df[df['Date'] >= start_date]
df_open_sell = df_filtered[df_filtered['Side'] == 'open sell']
df_close_buy = df_filtered[df_filtered['Side'] == 'close buy']
df_short = df_filtered[df_filtered['Side'] == 'short']

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the trade positions as scatter points
ax.scatter(df_open_sell['Date'], df_open_sell['Price'], marker='o', color='red', label='Open Sell', s=30, alpha=0.7)
ax.scatter(df_close_buy['Date'], df_close_buy['Price'], marker='o', color='blue', label='Close Buy', s=30, alpha=0.7)
ax.scatter(df_short['Date'], df_short['Price'], marker='o', color='green', label='Short', s=30, alpha=0.7)

# Plot the "price" as a line
ax.plot(df_filtered['Date'], df_filtered['Price'], color='orange', label='Price', linewidth=2)

# Set plot labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Trade Positions')

# Customize tick labels
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='both', labelsize=10)

# Add a grid
ax.grid(True, linestyle='--', alpha=0.5)

# Add legend with a shadow
ax.legend(frameon=True, shadow=True)

# Remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust plot margins
plt.tight_layout()

# Show the plot
plt.show()

# Filter the dataframe for the desired start date and actions
df_filtered = df[df['Date'] >= start_date]

# Plot the equity changes over time
plt.figure(figsize=(15, 6))
plt.plot(df_filtered['Date'], df_filtered['Equity'])
plt.title('Equity Changes over Time')
plt.xlabel('Date')
plt.ylabel('Equity')
plt.xticks(rotation=45)
plt.show()

# Define the URL of the endpoint
url = "https://test.deribit.com/api/v2/private/get_transaction_log"

start_date = pd.Timestamp(2023, 7, 9, 10, 11, 0)

# Start date can be set individually (e.g., 9 days before today)
start_date = int((datetime.now(timezone.utc) - timedelta(days=9)).timestamp() * 1e3) # Required type: integer
end_date = int(datetime.utcnow().timestamp() * 1e3)  # End date is set to the current timestamp

# Define the parameters
params = {
    "currency": "BTC",
    "start_timestamp": start_date,
    "end_timestamp": end_date
}

# Send the GET request
response = requests.get(url, params=params, auth=(ClientID, ClientSecret))
json_response = response.json()
data = json_response['result']

# Convert results to a DataFrame
df = json_normalize(data, 'logs')

# Filter for settlements only
df_settlement = df[df['type'] == 'settlement']





