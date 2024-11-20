# Create an abstract base class for lib_backtest strategies
# Each strategy should inherit from this class and implement the abstract methods

from abc import ABC, abstractmethod
from typing import List
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import os


class CommonUtilsBacktest:
    """
    Common utilities for backtesting
    """

    def __init__(self):
        pass

    def clean_df(self, df):
        print("===Cleaning DF===")
        print(f"Original shape: {df.shape}")
        df = df.dropna()
        print(f"After dropping NaNs: {df.shape}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us")
        df["local_timestamp"] = pd.to_datetime(df["local_timestamp"], unit="us")
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


@dataclass
class TradingPair:
    name: str
    amount: float


@dataclass
class BacktestInterfaceL1(ABC):
    @abstractmethod
    def __init__(self, source_dir, pair, starting_pair):
        self.bt_utils = CommonUtilsBacktest()

        trades_path = os.path.join(source_dir, "trades.csv")
        quotes_path = os.path.join(source_dir, "quotes.csv")

        for path in [trades_path, quotes_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File {path} not found")

        try:
            trades_df = pd.read_csv(source_dir + "trades.csv")
            quotes_df = pd.read_csv(source_dir + "quotes.csv")
        except pd.errors.EmptyDataError:
            raise ValueError("Empty CSV file")
        except pd.errors.ParserError:
            raise ValueError("Error parsing CSV file")
        except Exception as e:
            raise RuntimeError(f"Error reading CSV file: {e}")

        self.quotes = self.bt_utils.clean_df(quotes_df)
        self.quote_sample = self.quotes

        self.trades = self.bt_utils.clean_df(trades_df)
        self.trades_sample = self.trades

        self.buy_orders = []  # Outstanding orders
        self.sell_orders = []  # (price, amount, timestamp)
        self.filled_orders = []  # (price, amount, timestamp, direction)

        self.pair1 = TradingPair(pair[0], starting_pair[0])
        self.pair2 = TradingPair(pair[1], starting_pair[1])

    def sample_quotes(self, start_time="", end_time="", head=None, tail=None):
        self.quote_sample = self.bt_utils.sample_data(
            self.quotes, start_time, end_time, head, tail
        )

    def sample_trades(self, start_time="", end_time="", head=None, tail=None):
        self.trades_sample = self.bt_utils.sample_data(
            self.trades, start_time, end_time, head, tail
        )

    def trade_pair(self, direction, price, amount, fee=0.005):
        if direction == "buy":
            self.pair1.amount += amount
            self.pair2.amount -= price * amount * (1 + fee)
        elif direction == "sell":
            self.pair1.amount -= amount
            self.pair2.amount += price * amount * (1 - fee)
        else:
            raise ValueError("Invalid transaction direction")

    def place_order(self, direction, price, amount, timestamp):
        if direction == "buy":
            self.buy_orders.append((price, amount, timestamp))
        elif direction == "sell":
            self.sell_orders.append((price, amount, timestamp))
        else:
            raise ValueError("Invalid transaction direction")

    def modify_order(self, direction, price, amount, timestamp):
        pass  # Implement this

    def cancel_order(self, direction, price, amount, timestamp):
        pass  # Implement this

    def fill_orders(self, bid_price, ask_price):
        # Fill orders if the price is right
        # To be called at each time step
        for order in self.buy_orders:
            if order[0] >= ask_price:
                self.trade_pair("buy", order[0], order[1])
                self.filled_orders.append((order[0], order[1], order[2], "buy"))
                self.buy_orders.remove(order)
        for order in self.sell_orders:
            if order[0] <= bid_price:
                self.trade_pair("sell", order[0], order[1])
                self.filled_orders.append((order[0], order[1], order[2], "sell"))
                self.sell_orders.remove(order)

    @abstractmethod
    def backtest(self):
        pass


class BacktestInterfaceL2(ABC):
    @abstractmethod
    def __init__(self):
        self.bt_utils = CommonUtilsBacktest()
