from kfp.components import OutputPath


def get_data(output_path: OutputPath()):
    import os

    import yfinance as yf
    from pandas_datareader import data as pdr

    ticker = os.getenv("TICKER", "IBM")
    start_date = os.getenv("START_DATE", "2023-01-01")
    end_date = os.getenv("END_DATE", "2023-06-01")

    print(f"Ticker: {ticker}")

    yf.pdr_override()
    df = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)

    print(f"Count: \n{df.count()}")

    df.to_csv(output_path)

