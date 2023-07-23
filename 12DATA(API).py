import MetaTrader5 as mt5
import requests
import pandas as pd

def get_twelve_data(api_key, symbol, start_date, end_date):
    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1d",
        "start_date": start_date,
        "end_date": end_date,
        "apikey": api_key
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    if data.get("status", None) == "ok":
        df = pd.DataFrame(data["values"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        return df
    else:
        print("Error fetching data from Twelve Data API:", data.get("message"))
        return None

def main():
    # Replace your Twelve Data API key here
    twelve_data_api_key = "bee2a5b68c7f4f54b56f45a18206cc06"

    # Replace the symbol, start_date, and end_date with your desired stock and date range
    symbol = 'AAPL' , 'Nasdaq' , 'Dow_jones'
    start_date = '2023-07-22'
    end_date = '2025-12-19'

    # Get data from Twelve Data API
    data = get_twelve_data(twelve_data_api_key, symbol, start_date, end_date)

    if data is not None:
        print("Data fetched from Twelve Data API:")
        print(data)

        # Initialize MetaTrader 5
        mt5.initialize()

        # Transfer data to MetaTrader 5
        rates = []
        for index, row in data.iterrows():
            rate = (int(index.timestamp()), row['open'], row['high'], row['low'], row['close'], row['volume'])
            rates.append(rate)

        mt5.copy_rates_from(symbol, mt5.TIMEFRAME_D1, rates)

        # Shutdown MetaTrader 5
        mt5.shutdown()

if __name__ == "__main__":
    main()
