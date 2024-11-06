from tardis_dev import datasets

datasets.download(
    exchange="bybit",
    data_types=[
        "incremental_book_L2",
        "trades",
        "quotes",
        "derivative_ticker",
        "book_snapshot_25",
        "liquidations"
    ],
    from_date="2024-06-01",
    to_date="2024-06-02",
    symbols=["1000BEERUSDT"],
    # api_key="YOUR API KEY (optionally)",
)