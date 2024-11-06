# Create an abstract base class for lib_backtest strategies
# Each strategy should inherit from this class and implement the abstract methods

from abc import ABC, abstractmethod
from typing import List
import pandas as pd

class CommonUtilsBacktest():
    """
    Common utilities for backtesting
    """
    def __init__(self):
        pass

    def clean_df(self, df):
        print("===Cleaning DF===")
        print(f'Original shape: {df.shape}')
        df = df.dropna()
        print(f'After dropping NaNs: {df.shape}')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
        df['local_timestamp'] = pd.to_datetime(df['local_timestamp'], unit='us')
        return df

    def split_df(self, df, split_ratio):
        """
        Split the dataframe into training and testing data
        """
        split = int(split_ratio * len(df))
        train = df.iloc[:split]
        test = df.iloc[split:]
        return train, test

    def sample_data(self, df, start_time="", end_time="", head=None, tail=None):
        if start_time:
            df = df[df["timestamp"] >= start_time]
        if end_time:
            df = df[df["timestamp"] <= end_time]
        if head:
            df = df.head(head)
        if tail:
            df = df.tail(tail)
        return df



class BacktestInterfaceL1(ABC):
    @abstractmethod
    def __init__(self, source_dir, pair):
        self.bt_utils = CommonUtilsBacktest()

        trades_df = pd.read_csv(source_dir + "trades.csv")
        quotes_df = pd.read_csv(source_dir + "quotes.csv")

        self.quotes = self.bt_utils.clean_df(quotes_df)
        self.quote_sample = self.quotes

        self.trades = self.bt_utils.clean_df(trades_df)
        self.trades_sample = self.trades

        self.buy_orders = [] # Outstanding orders
        self.sell_orders = [] # (price, amount, timestamp)
        self.filled_orders = [] # (price, amount, timestamp, direction)

        self.pair1 = 0
        self.pair2 = 0

    def sample_quotes(self, start_time="", end_time="", head=None, tail=None):
        self.quote_sample = self.bt_utils.sample_data(self.quotes, start_time, end_time, head, tail)

    def sample_trades(self, start_time="", end_time="", head=None, tail=None):
        self.trades_sample = self.bt_utils.sample_data(self.trades, start_time, end_time, head, tail)





class BacktestInterfaceL2(ABC):
    @abstractmethod
    def __init__(self):
        self.bt_utils = CommonUtilsBacktest()





















