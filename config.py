
"""
Configuration Class holding constant variables
"""
class Config:
    # Stock information
    stock_name = 'FB'
    from_date = '2012-01-01'
    to_date = '2019-07-31'

    # Read transformed datasets from local machine
    read_from_local = True

    max_headline_length = 50
    max_daily_length = 200
    embedding_dim = 300

    # Neural model constants
    batch_size = 128
    epochs = 100
    validation_split = 0.15

