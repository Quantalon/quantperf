__version__ = '1.0.3'

from datetime import datetime
from functools import cached_property

import pandas as pd
import numpy as np


def rebase(prices: pd.Series):
    return prices / prices.iloc[0]


def calc_returns(prices: pd.Series):
    returns = prices.pct_change()
    return returns


def calc_eow_returns(returns: pd.Series):
    eow_returns = returns.groupby(
        pd.Grouper(freq='W')
    ).apply(lambda r: r.add(1).prod() - 1)
    return eow_returns


def calc_eom_returns(returns: pd.Series):
    eom_returns = returns.groupby(
        pd.Grouper(freq='M')
    ).apply(lambda r: r.add(1).prod() - 1)
    return eom_returns


def calc_eoy_returns(returns: pd.Series):
    eoy_returns = returns.groupby(
        pd.Grouper(freq='Y')
    ).apply(lambda r: r.add(1).prod() - 1)
    return eoy_returns


def calc_adj_returns(returns: pd.Series, rf=None):
    periods = 252
    if rf is None:
        return returns, periods
    daily_rf = np.power(1.0+rf, 1.0/periods) - 1
    returns_adj = returns - daily_rf
    return returns_adj, periods


def calc_total_return(prices: pd.Series):
    total_return = prices.iloc[-1] / prices.iloc[0] - 1
    return total_return


def calc_drawdowns(prices: pd.Series):
    roll_max = prices.cummax()
    drawdown = prices / roll_max - 1.0
    return drawdown


def calc_returns_mean(returns: pd.Series):
    periods = 252
    return returns.mean() * periods


def calc_volatility(returns: pd.Series):
    periods = 252
    vol = returns.std() * np.sqrt(periods)
    return vol


def calc_cagr(prices: pd.Series):
    total = prices.iloc[-1] / prices.iloc[0]
    years = (prices.index[-1] - prices.index[0]).days / 365.
    cagr = np.power(total, 1.0/years) - 1
    return cagr


def calc_sharpe(returns: pd.Series, rf=None):
    returns_adj, periods = calc_adj_returns(returns, rf)
    res = returns_adj.mean() / returns.std()
    sharpe = res * np.sqrt(periods)
    return sharpe


def calc_sortino(returns: pd.Series, rf=.0):
    returns_adj, periods = calc_adj_returns(returns, rf)
    downside_std = np.sqrt(np.square(np.minimum(returns_adj, 0.0)).mean())
    res = returns_adj.mean() / downside_std
    sortino = res * np.sqrt(periods)
    return sortino


def calc_mtd_return(returns: pd.Series):
    t = returns.index[-1]
    returns_mtd = returns[returns.index >= datetime(t.year, t.month, 1)]
    r = (returns_mtd+1).prod() - 1
    return r


def calc_1m_return(returns: pd.Series):
    s = returns.index[-1] - pd.DateOffset(months=1)
    returns = returns[returns.index >= s]
    r = (returns+1).prod() - 1
    return r


def calc_3m_return(returns: pd.Series):
    s = returns.index[-1] - pd.DateOffset(months=3)
    returns = returns[returns.index >= s]
    r = (returns+1).prod() - 1
    return r


def calc_6m_return(returns: pd.Series):
    s = returns.index[-1] - pd.DateOffset(months=6)
    returns = returns[returns.index >= s]
    r = (returns+1).prod() - 1
    return r


def calc_ytd_return(returns: pd.Series):
    t = returns.index[-1]
    returns = returns[returns.index >= datetime(t.year, 1, 1)]
    r = (returns+1).prod() - 1
    return r


def calc_1y_return(returns: pd.Series):
    s = returns.index[-1] - pd.DateOffset(years=1)
    returns = returns[returns.index >= s]
    r = (returns+1).prod() - 1
    return r


def calc_3y_return(returns: pd.Series):
    s = returns.index[-1] - pd.DateOffset(years=3)
    returns = returns[returns.index >= s]
    total = (returns+1).prod()
    years = (returns.index[-1] - returns.index[0]).days / 365.
    cagr = np.power(total, 1.0/years) - 1
    return cagr


def calc_5y_return(returns: pd.Series):
    s = returns.index[-1] - pd.DateOffset(years=5)
    returns = returns[returns.index >= s]
    total = (returns+1).prod()
    years = (returns.index[-1] - returns.index[0]).days / 365.
    cagr = np.power(total, 1.0/years) - 1
    return cagr


def calc_monthly_returns(returns: pd.Series):
    eom_returns = calc_eom_returns(returns)
    eom_returns.name = 'returns'
    eom_returns = eom_returns.to_frame()
    eom_returns['year'] = eom_returns.index.strftime('%Y')
    eom_returns['month'] = eom_returns.index.strftime('%m')
    monthly_returns = eom_returns.pivot('year', 'month', 'returns').fillna(0)

    month_columns = [f'{m:02d}' for m in range(1, 13)]
    # fulfill missing months
    for month in month_columns:
        if month not in monthly_returns.columns:
            monthly_returns.loc[:, month] = 0.0
    # order
    monthly_returns = monthly_returns[month_columns]

    return monthly_returns


def calc_max_drawdown(prices: pd.Series):
    max_drawdown = (prices / prices.cummax()).min() - 1
    return max_drawdown


def calc_drawdown_details(drawdowns: pd.Series):
    is_dd = drawdowns.ne(0)

    starts = is_dd & (~is_dd).shift(1)
    starts = starts[starts].index.to_list()

    ends = ~is_dd & is_dd.shift(1)
    ends = ends[ends].index.to_list()

    if len(starts) == 0:
        return None

    if starts[-1] > ends[-1]:
        ends.append(drawdowns.index[-1])

    data = []
    for i, _ in enumerate(starts):
        _drawdowns = drawdowns[starts[i]:ends[i]]
        data.append((
            starts[i].strftime('%Y-%m-%d'),
            ends[i].strftime('%Y-%m-%d'),
            (ends[i] - starts[i]).days,
            _drawdowns.min()
        ))

    details = pd.DataFrame(
        data,
        columns=['start', 'end', 'days', 'drawdown']
    )
    details = details.sort_values('drawdown')
    details = details.reset_index(drop=True)
    details.index.name = ''
    return details


def calc_drawdown_stats(drawdown_details: pd.DataFrame):
    stats = {}
    stats['avg_drawdown'] = drawdown_details['drawdown'].mean()
    stats['avg_drawdown_days'] = drawdown_details['days'].mean()
    stats['longest_drawdown_days'] = drawdown_details['days'].max()
    stats['max_drawdown'] = drawdown_details['days'].min()
    return stats


class Metrics:
    def __init__(self, prices: pd.Series, rf=None):
        self.prices = prices
        self.rf = rf
        self.start_time = self.prices.index[0]
        self.end_time = self.prices.index[-1]

    @cached_property
    def returns(self):
        return calc_returns(self.prices)

    @cached_property
    def drawdowns(self):
        return calc_drawdowns(self.prices)

    @cached_property
    def drawdown_details(self):
        return calc_drawdown_details(self.drawdowns)

    @cached_property
    def eow_returns(self):
        return calc_eow_returns(self.returns)

    @cached_property
    def eom_returns(self):
        return calc_eom_returns(self.returns)

    @cached_property
    def eoy_returns(self):
        return calc_eoy_returns(self.returns)

    @cached_property
    def monthly_returns(self):
        return calc_monthly_returns(self.returns)

    @cached_property
    def total_return(self):
        return calc_total_return(self.prices)

    @cached_property
    def cagr(self):
        return calc_cagr(self.prices)

    @cached_property
    def sharpe(self):
        return calc_sharpe(self.returns, self.rf)

    @cached_property
    def sortino(self):
        return calc_sortino(self.returns, self.rf)

    @cached_property
    def max_drawdown(self):
        return calc_max_drawdown(self.prices)

    @cached_property
    def longest_drawdown_days(self):
        return self.drawdown_details['days'].max()

    @cached_property
    def avg_drawdown(self):
        return self.drawdown_details['drawdown'].mean()

    @cached_property
    def avg_drawdown_days(self):
        return self.drawdown_details['days'].mean()

    @cached_property
    def volatility(self):
        return calc_volatility(self.returns)

    @cached_property
    def calmar(self):
        return self.cagr / abs(self.max_drawdown)

    @cached_property
    def skew(self):
        return self.returns.skew()

    @cached_property
    def kurt(self):
        return self.returns.kurt()

    @cached_property
    def mtd_return(self):
        return calc_mtd_return(self.returns)

    @cached_property
    def one_month_return(self):
        return calc_1m_return(self.returns)

    @cached_property
    def three_month_return(self):
        return calc_3m_return(self.returns)

    @cached_property
    def six_month_return(self):
        return calc_6m_return(self.returns)

    @cached_property
    def ytd_return(self):
        return calc_ytd_return(self.returns)

    @cached_property
    def one_year_return(self):
        return calc_1y_return(self.returns)

    @cached_property
    def three_year_return(self):
        return calc_3y_return(self.returns)

    @cached_property
    def best_day(self):
        return self.returns.max()

    @cached_property
    def worst_day(self):
        return self.returns.min()

    @cached_property
    def best_week(self):
        return self.eow_returns.max()

    @cached_property
    def worst_week(self):
        return self.eow_returns.min()

    @cached_property
    def best_month(self):
        return self.eom_returns.max()

    @cached_property
    def worst_month(self):
        return self.eom_returns.min()

    @cached_property
    def win_rate_day(self):
        win_count = self.returns[self.returns > 0].count()
        return win_count / self.returns.count()

    @cached_property
    def win_rate_week(self):
        win_count = self.eow_returns[self.eow_returns > 0].count()
        return win_count/self.eow_returns.count()

    @cached_property
    def win_rate_month(self):
        win_count = self.eom_returns[self.eom_returns > 0].count()
        return win_count / self.eom_returns.count()

    @cached_property
    def stats(self):
        _r = [
            ('Start Time', self.start_time.strftime('%Y-%m-%d')),
            ('End Time', self.end_time.strftime('%Y-%m-%d')),
            ('Risk-Free Rate', '-' if self.rf is None else f'{self.rf:.2%}'),
            ('Total Return', f'{self.total_return:.2%}'),
            ('CAGR', f'{self.cagr:.2%}'),
            ('Sharpe', f'{self.sharpe:.2f}'),
            ('Sortino', f'{self.sortino:.2f}'),
            ('Max Drawdown', f'{self.max_drawdown:.2%}'),
            ('Longest Drawdown Days', f'{self.longest_drawdown_days}'),
            ('Avg Drawdown', f'{self.avg_drawdown:.2%}'),
            ('Avg Drawdown Days', f'{self.avg_drawdown_days:.2f}'),
            ('Volatility', f'{self.volatility:.2%}'),
            ('Calmar', f'{self.calmar:.2f}'),
            ('Skew', f'{self.skew:.2f}'),
            ('Kurtosis', f'{self.kurt:.2f}'),
            ('MTD', f'{self.mtd_return:.2%}'),
            ('3M', f'{self.three_month_return:.2%}'),
            ('6M', f'{self.six_month_return:.2%}'),
            ('YTD', f'{self.ytd_return:.2%}'),
            ('1Y', f'{self.one_year_return:.2%}'),
            ('3Y', f'{self.three_year_return:.2%}'),
            ('Best Day', f'{self.best_day:.2%}'),
            ('Worst Day', f'{self.worst_day:.2%}'),
            ('Best Week', f'{self.best_week:.2%}'),
            ('Worst Week', f'{self.worst_week:.2%}'),
            ('Best Month', f'{self.best_month:.2%}'),
            ('Worst Month', f'{self.worst_month:.2%}'),
            ('Win% Day', f'{self.win_rate_day:.2%}'),
            ('Win% Week', f'{self.win_rate_week:.2%}'),
            ('Win% Month', f'{self.win_rate_month:.2%}'),
        ]
        s = pd.Series(dict(_r), name=self.prices.name)
        s.index.name = ''
        return s
