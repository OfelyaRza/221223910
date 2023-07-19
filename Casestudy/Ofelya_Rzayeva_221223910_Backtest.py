conda activate case-env

import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import requests

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

url = 'https://test.deribit.com/api/v2/'

class DataHandler:
    def __init__(self, db_name):
        self.engine = create_engine(f'sqlite:///{db_name}')
        self.url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
        
    def download(self, instrument_name):
        msg = {
                "jsonrpc" : "2.0",
                "id" : 833,
                "method" : "public/get_tradingview_chart_data",
                "params" : {
                "instrument_name" : instrument_name,
                "end_timestamp": int(time.time() * 1000),  # In Epoch milliseconds
                "start_timestamp": int((time.time() - 1e6) * 1000), # In Epoch milliseconds
                "resolution" : "60"  # Minute data
                }
            }
        response = requests.post(url, json=msg)
        data = response.json()
        ohlc = pd.DataFrame(data['result'])
        ohlc['timestamp'] = pd.to_datetime(ohlc['ticks'], unit='ms')
        ohlc['instrument_name'] = instrument_name
        ohlc['resolution'] = 60
        # Please note the if_exists='replace' ... one might if_exists='append' but need to check for duplicates!
        ohlc.to_sql('ohlc', self.engine, if_exists='replace')
        
    def select(self, query):
        return pd.read_sql(query, self.engine)
        
    def plot(self, query):
        df = self.select(query)
        df.plot()
        plt.title(f'{query}')
        plt.show()

# Download OHLC data for the instrument
dh = DataHandler('07_datam.db')
dh.download('BTC-PERPETUAL')
data = dh.select('SELECT * FROM ohlc')
dh.plot('SELECT timestamp, close FROM ohlc ORDER BY timestamp ASC')

# Calculate breakout prices
instrument_name = "BTC-PERPETUAL"
threshold_percentage = 2.0

# Get the current market price
market_price = get_market(instrument_name)

# Calculate breakout threshold
threshold = market_price * threshold_percentage / 100

# Determine breakout prices
breakout_high = market_price + threshold
breakout_low = market_price - threshold

print("Breakout prices:")
print(f"Breakout High: {breakout_high}")
print(f"Breakout Low: {breakout_low}")

# Load the data from the database
engine = create_engine('sqlite:///07_datam.db')
ohlc_data = pd.read_sql("SELECT * FROM ohlc", engine)
ohlc_data['timestamp'] = pd.to_datetime(ohlc_data['timestamp'])  # Convert 'timestamp' to datetime format

# Calculate daily log returns
ohlc_data['Returns'] = np.log(ohlc_data['close'] / ohlc_data['close'].shift(1))

# Initialize the 'Breakout_Signal' column with 0
ohlc_data['Breakout_Signal'] = 0

# Generate the signal for the Breakout strategy
ohlc_data.loc[ohlc_data['close'] > breakout_high, 'Breakout_Signal'] = -1
ohlc_data.loc[ohlc_data['close'] < breakout_low, 'Breakout_Signal'] = 1

# Backfill any remaining NaN values in the 'Breakout_Signal' column
ohlc_data['Breakout_Signal'] = ohlc_data['Breakout_Signal'].fillna(method='ffill')

# Calculate the strategy returns for the Breakout strategy using the current day's 'close' prices
ohlc_data['Breakout_Strategy_Returns'] = ohlc_data['Returns'] * ohlc_data['Breakout_Signal'].shift(1).fillna(0)

# Calculate cumulative returns for the Breakout strategy
ohlc_data['Breakout_Cumulative_Returns'] = (ohlc_data['Breakout_Strategy_Returns'] + 1).cumprod() - 1

# Calculate RSI Strategy
window = 14
delta = ohlc_data['close'].diff()  # Calculate the price change between each period
gain = delta.where(delta > 0, 0)  # Calculate the gain by setting negative values to 0
loss = -delta.where(delta < 0, 0)  # Calculate the loss by setting positive values to 0
average_gain = gain.rolling(window).mean()  # Calculate the average gain over the specified window
average_loss = loss.rolling(window).mean()  # Calculate the average loss over the specified window
relative_strength = average_gain / average_loss  # Calculate the relative strength
rsi = 100 - (100 / (1 + relative_strength))  # Calculate the RSI

# Generate the signal for the RSI strategy
ohlc_data['RSI_Signal'] = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))

# Calculate the strategy returns for the RSI strategy using the previous day's 'close' prices
ohlc_data['RSI_Strategy_Returns'] = ohlc_data['Returns'] * ohlc_data['RSI_Signal'].shift(1).fillna(0)

# Calculate cumulative returns for the RSI strategy
ohlc_data['RSI_Cumulative_Returns'] = (ohlc_data['RSI_Strategy_Returns'] + 1).cumprod() - 1

# Calculate Bollinger Bands
window = 20
ohlc_data['Rolling_Mean'] = ohlc_data['close'].rolling(window=window).mean()
ohlc_data['Rolling_Std'] = ohlc_data['close'].rolling(window=window).std()

# Calculate upper and lower Bollinger Bands
ohlc_data['Upper_Band'] = ohlc_data['Rolling_Mean'] + 2 * ohlc_data['Rolling_Std']
ohlc_data['Lower_Band'] = ohlc_data['Rolling_Mean'] - 2 * ohlc_data['Rolling_Std']

# Generate the signal for the Bollinger Bands strategy
ohlc_data['Bollinger_Bands_Signal'] = np.where(ohlc_data['close'] < ohlc_data['Lower_Band'], 1,
                                               np.where(ohlc_data['close'] > ohlc_data['Upper_Band'], -1, 0))

# Calculate the strategy returns for the Bollinger Bands strategy using the previous day's 'close' prices
ohlc_data['Bollinger_Bands_Strategy_Returns'] = ohlc_data['Returns'] * ohlc_data['Bollinger_Bands_Signal'].shift(1).fillna(0)

# Calculate cumulative returns for the Bollinger Bands strategy
ohlc_data['Bollinger_Bands_Cumulative_Returns'] = (ohlc_data['Bollinger_Bands_Strategy_Returns'] + 1).cumprod() - 1

# Calculate MACD Strategy
short_window = 12
long_window = 26

# Calculate short-term and long-term exponential moving averages
short_ema = ohlc_data['close'].ewm(span=short_window, adjust=False).mean()
long_ema = ohlc_data['close'].ewm(span=long_window, adjust=False).mean()

# Calculate the MACD line and the signal line
macd_line = short_ema - long_ema
signal_line = macd_line.ewm(span=9, adjust=False).mean()

# Generate the signal for the MACD strategy
ohlc_data['MACD_Signal'] = np.where(macd_line > signal_line, 1, -1)

# Calculate the strategy returns for the MACD strategy using the previous day's 'close' prices
ohlc_data['MACD_Strategy_Returns'] = ohlc_data['Returns'] * ohlc_data['MACD_Signal'].shift(1).fillna(0)

# Calculate cumulative returns for the MACD strategy
ohlc_data['MACD_Cumulative_Returns'] = (ohlc_data['MACD_Strategy_Returns'] + 1).cumprod() - 1

# Calculate Single Moving Average (SMA) Strategy
lookback_period = 10
ohlc_data['SMA'] = ohlc_data['close'].rolling(window=lookback_period).mean()

# Generate the signal for the SMA strategy
ohlc_data['SMA_Signal'] = np.where(ohlc_data['close'] >= ohlc_data['SMA'], 1, -1)

# Backtest the SMA strategy
ohlc_data['SMA_Strategy_Returns'] = ohlc_data['Returns'] * ohlc_data['SMA_Signal'].shift(1).fillna(0)

# Calculate cumulative returns for the SMA strategy
ohlc_data['SMA_Cumulative_Returns'] = (ohlc_data['SMA_Strategy_Returns'] + 1).cumprod() - 1

# Plot the cumulative returns for all four strategies in the same plot
plt.figure(figsize=(20, 8))
plt.plot(ohlc_data['timestamp'], ohlc_data['SMA_Cumulative_Returns'], label='Single Moving Average', color='blue')
plt.plot(ohlc_data['timestamp'], ohlc_data['RSI_Cumulative_Returns'], label='RSI', color='green')
plt.plot(ohlc_data['timestamp'], ohlc_data['Breakout_Cumulative_Returns'], label='Breakout Strategy', color='red')
plt.plot(ohlc_data['timestamp'], ohlc_data['Bollinger_Bands_Cumulative_Returns'], label='Bollinger Bands', color='purple')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.title('Cumulative Returns of Different Strategies')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


conda deactivate
