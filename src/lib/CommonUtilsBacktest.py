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



class BacktestInterface(ABC):
    @abstractmethod
    def __init__(self):
        self.bt_utils = CommonUtilsBacktest()

