# QuantPerf

## QuickStart

```bash
# pip install quantperf
```

```python
import yfinance as yf
from quantperf import Metrics

daily_prices = yf.Ticker("AAPL").history()['Close']

# The input must be **daily** prices
metrics = Metrics(daily_prices)

print(metrics.stats)
```