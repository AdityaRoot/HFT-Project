import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import abstractmethod

from CommonUtilsBacktest import BacktestInterfaceL1


class ModelBEERUSDT(BacktestInterfaceL1):
    def __init__(
        self,
        source_dir,
        starting_usd=1000,
    ):
        super().__init__(source_dir, ('beer', 'usdt'), ())

        self.pair1 = starting_usd
        self.beer = 0


    def trade_pair(self, direction, execution_price, amount, fee=0.005):
        if direction == "buy":
            self.beer += amount
            self.usdt -= execution_price * amount * (1 + fee)
        elif direction == "sell":
            self.beer -= amount
            self.usdt += execution_price * amount * (1 - fee)
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
            x="timestamp",
            y="ask_price",
            data=self.quote_sample,
            ax=ax,
            label="Ask Price",
        )
        sns.lineplot(
            x="timestamp",
            y="bid_price",
            data=self.quote_sample,
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
            label="Trades",
        )
        plt.show()

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            x="timestamp",
            y="ask_price",
            data=self.quote_sample,
            ax=ax,
            label="Ask Price",
        )
        sns.lineplot(
            x="timestamp",
            y="bid_price",
            data=self.quote_sample,
            ax=ax,
            label="Bid Price",
        )

        # Color by 'side' column
        sns.scatterplot(
            x="timestamp",
            y="price",
            data=self.trades_sample,
            ax=ax,
            label="Trades",
            hue="side",
        )
        plt.show()

    @abstractmethod
    def backtest(self):
        pass


if __name__ == "__main__":
    print('If mained')
    input_csv = "../data/quotes.csv"
    sample_df = pd.read_csv(input_csv)
    sample_df = sample_df.head(100)

    print(sample_df)
    model = ModelBEERUSDT(quote_df=sample_df, starting_usd=1000)
    model.plot_quotes()
    # model.lib_backtest()
