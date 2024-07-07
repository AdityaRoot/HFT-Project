import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.backtest.CommonUtilsBacktest import BacktestInterface


class BaselineModel(BacktestInterface):
    def __init__(
        self,
        quote_df,
        starting_usd=1000,
    ):
        super().__init__()
        self.usdt = starting_usd
        self.beer = 0

        self.quotes = self.bt_utils.clean_df(quote_df)
        self.quote_sample = self.quotes

        self.buy_order = []
        self.sell_order = []

    def clean_quotes(self):
        print("===Cleaning DF===")
        print(f"Original shape: {self.quotes.shape}")
        self.quotes = self.quotes.dropna()
        print(f"After dropping NaNs: {self.quotes.shape}")

        self.quotes["timestamp"] = pd.to_datetime(self.quotes["timestamp"], unit="us")
        self.quotes["local_timestamp"] = pd.to_datetime(
            self.quotes["local_timestamp"], unit="us"
        )

    def sample_quotes(self, start_time="", end_time="", head=None, tail=None):
        if start_time:  # Cut by start time
            self.quote_sample = self.quotes[self.quotes["timestamp"] >= start_time]
        if end_time:
            self.quote_sample = self.quotes[self.quotes["timestamp"] <= end_time]
        if head:
            self.quote_sample = self.quotes.head(head)
        if tail:
            self.quote_sample = self.quotes.tail(tail)

    def buy(self, price, amount, fee=0.005):
        self.beer += amount
        self.usdt -= price * amount * (1 + fee)

    def sell(self, price, amount, fee=0.005):
        self.beer -= amount
        self.usdt += price * amount * (1 - fee)

    def plot_quotes(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            x="local_timestamp",
            y="ask_price",
            data=self.quote_sample,
            ax=ax,
            label="Ask Price",
        )
        sns.lineplot(
            x="local_timestamp",
            y="bid_price",
            data=self.quote_sample,
            ax=ax,
            label="Bid Price",
        )
        plt.show()

    # def

    def backtest(self):
        pass


if __name__ == "__main__":
    input_csv = "../data/bybit_quotes_2024-06-14_1000BEERUSDT.csv"
    sample_df = pd.read_csv(input_csv)
    sample_df = sample_df.head(100)

    print(sample_df)
    model = BaselineModel(quote_df=sample_df, starting_usd=1000)
    model.plot_quotes()
    # model.backtest()
