import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import abstractmethod

from CommonUtilsBacktest import BacktestInterface


class ModelBEERUSDT(BacktestInterface):
    def __init__(
        self,
        quote_df,
        trade_df,
        starting_usd=1000,
    ):
        super().__init__()
        self.usdt = starting_usd
        self.beer = 0

        self.quotes = self.bt_utils.clean_df(quote_df)
        self.quote_sample = self.quotes

        self.trades = self.bt_utils.clean_df(trade_df)
        self.trades_sample = self.trades

        self.buy_orders = [] # Outstanding orders
        self.sell_orders = [] # (price, amount, timestamp)
        self.filled_orders = [] # (price, amount, timestamp, direction)

    def sample_quotes(self, start_time="", end_time="", head=None, tail=None):
        self.bt_utils.sample_data(self.quotes, start_time, end_time, head, tail)

    def sample_trades(self, start_time="", end_time="", head=None, tail=None):
        self.bt_utils.sample_data(self.trades, start_time, end_time, head, tail)

    def trade_pair(self, direction, price, amount, fee=0.005):
        if direction == "buy":
            self.beer += amount
            self.usdt -= price * amount * (1 + fee)
        elif direction == "sell":
            self.beer -= amount
            self.usdt += price * amount * (1 - fee)
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
        pass # TO be called when low on funds, to cap the amount of beer to buy or sell

    def cancel_order(self, direction, price, amount, timestamp):
        pass

    def fill_orders(self, bid_price, ask_price):
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

    def plot_trades(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(
            x="local_timestamp",
            y="price",
            data=self.trades_sample,
            ax=ax,
            label="Trades",
        )
        plt.show()

    def plot(self):
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
        sns.scatterplot(
            x="local_timestamp",
            y="price",
            data=self.trades_sample,
            ax=ax,
            label="Trades",
        )
        plt.show()

    @abstractmethod
    def backtest(self):
        pass


if __name__ == "__main__":
    print('If mained')
    input_csv = "../data/bybit_quotes_2024-06-14_1000BEERUSDT.csv"
    sample_df = pd.read_csv(input_csv)
    sample_df = sample_df.head(100)

    print(sample_df)
    model = ModelBEERUSDT(quote_df=sample_df, starting_usd=1000)
    model.plot_quotes()
    # model.lib_backtest()
