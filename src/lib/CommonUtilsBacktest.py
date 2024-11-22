# Create an abstract base class for lib_backtest strategies
# Each strategy should inherit from this class and implement the abstract methods

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from bisect import bisect_left, bisect_right, bisect
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging


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



class OrderDirection(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass(order=True)
class Order:
    price: float
    amount: float
    timestamp: pd.Timestamp

@dataclass
class FilledOrder(Order):
    direction: OrderDirection

@dataclass
class TradingPair:
    name: str
    amount: float

@dataclass
class BacktestInterfaceL1(ABC):
    @abstractmethod
    def __init__(self, source_dir: str, pair: Tuple[str, str], starting_pair: Tuple[float, float]):
        self.bt_utils = CommonUtilsBacktest()

        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        trades_path = os.path.join(source_dir, "trades.csv.gz")
        quotes_path = os.path.join(source_dir, "quotes.csv.gz")
        for path in [trades_path, quotes_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File {path} not found")
        try:
            trades_df = pd.read_csv(source_dir + "trades.csv.gz")
            quotes_df = pd.read_csv(source_dir + "quotes.csv.gz")
        except pd.errors.EmptyDataError:
            raise ValueError("Empty CSV file")
        except pd.errors.ParserError:
            raise ValueError("Error parsing CSV file")
        except Exception as e:
            raise RuntimeError(f"Error reading CSV file: {e}")

        self.quotes = self.bt_utils.clean_df(quotes_df)
        self.quotes_sample = self.quotes
        self.trades = self.bt_utils.clean_df(trades_df)
        self.trades_sample = self.trades

        self.buy_orders = []  # Outstanding orders
        self.sell_orders = []  # (price, amount, timestamp)
        self.filled_orders = []  # (price, amount, timestamp, direction)

        self.pair1 = TradingPair(pair[0], starting_pair[0])
        self.pair2 = TradingPair(pair[1], starting_pair[1])

    def sample_quotes(self, start_time="", end_time="", head=None, tail=None):
        self.quotes_sample = self.bt_utils.sample_data(
            self.quotes, start_time, end_time, head, tail
        )

    def sample_trades(self, start_time="", end_time="", head=None, tail=None):
        self.trades_sample = self.bt_utils.sample_data(
            self.trades, start_time, end_time, head, tail
        )

    def trade_pair(self, direction: OrderDirection, execution_price: float, amount: float, fee: float = 0.005):
        if direction == OrderDirection.BUY:
            self.pair1.amount += amount
            self.pair2.amount -= execution_price * amount * (1 + fee)
        elif direction == OrderDirection.SELL:
            self.pair1.amount -= amount
            self.pair2.amount += execution_price * amount * (1 - fee)
        else:
            raise ValueError("Invalid transaction direction")
        self.logger.debug(f"{direction.value.capitalize()} {amount} {self.pair1.name} at {execution_price} {self.pair2.name}")

    def place_order(self, direction: OrderDirection, price: float, amount: float, timestamp: pd.Timestamp) -> None:
        order = Order(price, amount, timestamp)
        # Insert while maintaining order
        if direction == OrderDirection.BUY:
            index = bisect_left([-o.price for o in self.buy_orders], -order.price)
            self.buy_orders.insert(index, order)
        elif direction == OrderDirection.SELL:
            index = bisect_left([o.price for o in self.sell_orders], order.price)
            self.sell_orders.insert(index, order)
        else:
            raise ValueError("Invalid transaction direction")

    def modify_order(self, direction: OrderDirection, price: float, amount: float, timestamp: pd.Timestamp):
        pass  # Implement this

    def cancel_order(self, direction: OrderDirection, price: float, amount: float, timestamp: pd.Timestamp):
        pass  # Implement this

    def fill_orders(self, bid_price, ask_price):
        # Fill orders if the price is right
        # To be called at each time step
        for order in self.buy_orders[:]: # Iterate over a copy of the list to avoid modifying it while iterating
            if order.price >= ask_price:
                execution_price = ask_price
                self.trade_pair(OrderDirection.BUY, execution_price, order.amount)
                filled_order = FilledOrder(
                    price = execution_price,
                    amount = order.amount,
                    timestamp = order.timestamp,
                    direction = OrderDirection.BUY
                )
                self.filled_orders.append(filled_order)
                self.buy_orders.remove(order)
        for order in self.sell_orders[:]:
            if order.price <= bid_price:
                execution_price = bid_price
                self.trade_pair(OrderDirection.SELL, execution_price, order.amount)
                filled_order = FilledOrder(
                    price = execution_price,
                    amount = order.amount,
                    timestamp = order.timestamp,
                    direction = OrderDirection.SELL
                )
                self.filled_orders.append(filled_order)
                self.sell_orders.remove(order)

    @abstractmethod
    def backtest(self):
        pass

    def plot_quotes(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            x="timestamp",
            y="ask_price",
            data=self.quotes_sample,
            ax=ax,
            label="Ask Price",
        )
        sns.lineplot(
            x="timestamp",
            y="bid_price",
            data=self.quotes_sample,
            ax=ax,
            label="Bid Price",
        )
        plt.show()

    def plot_trades(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(
            x="timestamp",
            y="price",
            data=self.trades_sample,
            ax=ax,
            hue="direction",
        )
        plt.show()

    def plot(self):
        self.logger.info("Plotting quotes and trades")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            x="timestamp",
            y="ask_price",
            data=self.quotes_sample,
            ax=ax,
            label="Ask Price",
        )
        sns.lineplot(
            x="timestamp",
            y="bid_price",
            data=self.quotes_sample,
            ax=ax,
            label="Bid Price",
        )
        sns.scatterplot(
            x="timestamp",
            y="price",
            data=self.trades_sample,
            ax=ax,
            hue="side",
        )
        sns.scatterplot(
            x="timestamp",
            y="price",
            data=self.filled_orders,
            ax=ax,
            hue="direction",
        )

        plt.show()

    def plot_filled_orders(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(
            x="timestamp",
            y="price",
            data=self.filled_orders,
            ax=ax,
            hue="direction",
        )
        plt.show()



class BacktestInterfaceL2_snapshot(ABC):
    @abstractmethod
    def __init__(self):
        self.bt_utils = CommonUtilsBacktest()
