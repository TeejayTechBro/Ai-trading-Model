# Author : Rasheed Tijani 
# Date : 23 july , 2023

# Project : Analyzing market data with price correlation (Pearson correlation)

import pandas
import requests
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt 

start = dt.datetime(2023,7,20)
end = dt.datetime.now()

tickers = ["FB", "MSFT" , "NVDA" , "TSLA" , "AAPL", "NSDQ" , "DOW_JONES"]
colnames = []

for ticker in tickers:
    data = web.DataReader(ticker, "twelvedata", start , end)
    if len(colnames) == 0:
        combined = data[['Adj Close']].copy()
        colnames.append(ticker)
        combined.columns = colnames

else:
    combined = combined.join(data['Adj Close'])
    colnames.append(ticker)
    combined.columns = colnames
    
for ticker in tickers:
    plt.plot(loc = "upper right")
    plt.show()
    
corr_data = combined.pct_change().corr(method ="pearson")
sns.heatmap(corr_data, annot = True , cmap ="coolwarm")

plt.show()

                
#### correlation of market data and 12API data with MetaTRaDER 

def get_historical_data(symbols, start_date, end_date):
    api_key = 'bee2a5b68c7f4f54b56f45a18206cc06'
    data = {}
    for symbol in symbols:
        url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval=1d&start_date={start_date}&end_date={end_date}&apikey={api_key}'
        response = requests.get(url)
        if response.status_code == 200:
            data[symbol] = response.json()['values']
    return data

def calculate_pearson_correlation(data):
    df = pd.DataFrame(data)
    correlation_matrix = df.corr(method='pearson')
    return correlation_matrix

def main():
    # Replace with the desired symbols as a list (e.g., ['AAPL', 'MSFT', 'GOOG'])
    symbols = ['AAPL', 'MSFT', 'GOOG','TSLA','NSDQ','']
    
    # Replace with the desired start and end dates in 'YYYY-MM-DD' format
    start_date = '2022-01-01'
    end_date = '2023-01-01'

    # Fetch historical price data from 12Data API
    data = get_historical_data(symbols, start_date, end_date)

    if len(data) > 0:
        # Create a DataFrame with the fetched data
        df = pd.DataFrame(data)

        # Print the first few rows of the data
        print(df.head())

        # Calculate Pearson correlation matrix
        correlation_matrix = calculate_pearson_correlation(df)

        # Print Pearson correlation matrix
        print("Pearson Correlation Matrix:")
        print(correlation_matrix)
    else:
        print("No data available for the specified symbols or API request failed.")

if __name__ == "__main__":
    main()


###### Scanning Daily Market mover (open , high & close)

def scan_daily_market_movers(market, limit):
    api_key = 'bee2a5b68c7f4f54b56f45a18206cc06'
    url = f'https://api.twelvedata.com/top-movers?apikey={api_key}&market={market}&limit={limit}'
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['data']
    else:
        print(f"Failed to fetch data. Error Code: {response.status_code}")
        return []

def main():
    # Replace 'YOUR_12DATA_API_KEY' with your actual API key from 12Data.
    api_key = 'bee2a5b68c7f4f54b56f45a18206cc06'
    
    # Replace 'US' with the desired market (e.g., 'US', 'FOREX', 'CRYPTO', 'INDEX')
    market = 'US', 'European' , 'Tokyo' 
    
    # Set the limit for the number of top movers to fetch
    limit = 10
    
    # Scan daily market movers
    top_movers = scan_daily_market_movers(market, limit)

    if top_movers:
        for mover in top_movers:
            symbol = mover['symbol']
            percentage_change = mover['percentage_change']
            print(f"{symbol}: {percentage_change}%")
    else:
        print("No data available for the specified market or API request failed.")

if __name__ == "__main__":
    main()


