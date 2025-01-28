from numba import njit
from hftbacktest import GTX, LIMIT, Recorder, BacktestAsset, HashMapMarketDepthBacktest
from hftbacktest.stats import LinearAssetRecord
import numpy as np

asset = (
    BacktestAsset()
        .data(['hft_db/beerusdt_20240601.npz']) # No initial snapshot as data comes from tardis.dev
        .linear_asset(1.0)
        .constant_latency(10_000_000, 10_000_000)
        .log_prob_queue_model2() # Issue persists with risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(-0.00005, 0.00005)
        .tick_size(0.00001)
        .lot_size(0.1)
        .last_trades_capacity(0)
)


@njit
def submit_order_stats(hbt, recorder):
    buy_order_id = 1
    sell_order_id = 2
    half_spread = 5 * hbt.depth(0).tick_size


    while hbt.elapse(1 * 1e9) == 0:
        hbt.clear_inactive_orders(0)

        depth = hbt.depth(0)
        mid_price = (depth.best_bid + depth.best_ask) / 2.0

        if buy_order_id not in hbt.orders(0):
            order_price = round((mid_price - half_spread) / depth.tick_size) * depth.tick_size
            order_qty = 1
            time_in_force = GTX
            order_type = LIMIT
            hbt.submit_buy_order(0, buy_order_id, order_price, order_qty, time_in_force, order_type, False)
        else:
            hbt.cancel(0, buy_order_id, False)

        if sell_order_id not in hbt.orders(0):
            order_price = round((mid_price + half_spread) / depth.tick_size) * depth.tick_size
            order_qty = 1
            time_in_force = GTX
            order_type = LIMIT
            hbt.submit_sell_order(0, sell_order_id, order_price, order_qty, time_in_force, order_type, False)
        else:
            hbt.cancel(0, sell_order_id, False)

        recorder.record(hbt)
    return True

hbt = HashMapMarketDepthBacktest([asset])

recorder = Recorder(
    hbt.num_assets,
    1000000
)

submit_order_stats(hbt, recorder.recorder)

_ = hbt.close()

record = LinearAssetRecord(recorder.get(0))
stats = record.stats()
print(stats.summary())
#%%
# display(stats.plot())