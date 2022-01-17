# QuantPerf

## QuickStart

```bash
# pip install quantperf
```

```python
import yfinance as yf
from quantperf import Metrics

aapl = yf.Ticker("AAPL")
data = aapl.history()
prices = data['Close']
metrics = Metrics(prices)

print(metrics.stats)
```