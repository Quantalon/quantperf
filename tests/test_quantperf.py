from pathlib import Path

import pandas as pd

from quantperf import __version__
from quantperf import Metrics


def test_version():
    assert __version__ == '1.0.3'


def test_metrics():
    test_data_file = Path(__file__).parent / 'test_data_aapl.csv'
    df = pd.read_csv(test_data_file, parse_dates=['Date'], index_col='Date')
    prices = df['Close']
    metrics = Metrics(prices)
    stats = metrics.stats
    print(stats)