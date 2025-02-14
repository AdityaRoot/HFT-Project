from tardis_dev import datasets

datasets.download(
    exchange="binance",
    data_types=[
        "incremental_book_L2",
        "trades",
    ],
    from_date="2024-06-01",
    to_date="2024-06-02",
    symbols=["BTCUSDT"],
    # api_key="YOUR API KEY (optionally)",
)