__version__ = '0.1.3'

from datetime import datetime

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
    # handle missing months
    for month in month_columns:
        if month not in monthly_returns.columns:
            monthly_returns.loc[:, month] = 0.0
    # order columns by month
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
    def __init__(self, prices: pd.Series, benchmark: pd.Series = None, rf=None):

        self._prices = prices
        self._benchmark = benchmark
        self._rf = rf
        self._start_time = self._prices.index[0]
        self._end_time = self._prices.index[-1]

        # series
        self._returns = None
        self._drawdowns = None
        self._eow_returns = None
        self._eom_returns = None
        self._eoy_returns = None

        # details, pd.DataFrame
        self._drawdown_details = None
        self._monthly_returns = None

        # metrics
        self._total_return = None
        self._cagr = None
        self._sharpe = None
        self._sortino = None
        self._max_drawdown = None
        self._longest_drawdown_days = None
        self._avg_drawdown = None
        self._avg_drawdown_days = None
        self._volatility = None
        self._calmar = None
        self._skew = None
        self._kurt = None

        self._mtd_return = None
        self._one_month_return = None
        self._three_month_return = None
        self._six_month_return = None
        self._ytd_return = None
        self._one_year_return = None
        self._three_year_return = None

        self._best_day = None
        self._worst_day = None
        self._best_week = None
        self._worst_week = None
        self._best_month = None
        self._worst_month = None

        self._win_rate_day = None
        self._win_rate_week = None
        self._win_rate_month = None

    @property
    def rf(self):
        return self._rf

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def returns(self):
        if self._returns is None:
            self._returns = calc_returns(self._prices)
        return self._returns

    @property
    def drawdowns(self):
        if self._drawdowns is None:
            self._drawdowns = calc_drawdowns(self._prices)
        return self._drawdowns

    @property
    def drawdown_details(self):
        if self._drawdown_details is None:
            self._drawdown_details = calc_drawdown_details(self.drawdowns)
        return self._drawdown_details

    @property
    def eow_returns(self):
        if self._eow_returns is None:
            self._eow_returns = calc_eow_returns(self.returns)
        return self._eow_returns

    @property
    def eom_returns(self):
        if self._eom_returns is None:
            self._eom_returns = calc_eom_returns(self.returns)
        return self._eom_returns

    @property
    def eoy_returns(self):
        if self._eoy_returns is None:
            self._eoy_returns = calc_eoy_returns(self.returns)
        return self._eoy_returns

    @property
    def monthly_returns(self):
        if self._monthly_returns is None:
            self._monthly_returns = calc_monthly_returns(self.returns)
        return self._monthly_returns

    @property
    def total_return(self):
        if self._total_return is None:
            self._total_return = calc_total_return(self._prices)
        return self._total_return

    @property
    def cagr(self):
        if self._cagr is None:
            self._cagr = calc_cagr(self._prices)
        return self._cagr

    @property
    def sharpe(self):
        if self._sharpe is None:
            self._sharpe = calc_sharpe(self.returns, self.rf)
        return self._sharpe

    @property
    def sortino(self):
        if self._sortino is None:
            self._sortino = calc_sortino(self.returns, self.rf)
        return self._sortino

    @property
    def max_drawdown(self):
        if self._max_drawdown is None:
            self._max_drawdown = calc_max_drawdown(self._prices)
        return self._max_drawdown

    @property
    def longest_drawdown_days(self):
        if self._longest_drawdown_days is None:
            self._longest_drawdown_days = self.drawdown_details['days'].max()
        return self._longest_drawdown_days

    @property
    def avg_drawdown(self):
        if self._avg_drawdown is None:
            self._avg_drawdown = self.drawdown_details['drawdown'].mean()
        return self._avg_drawdown

    @property
    def avg_drawdown_days(self):
        if self._avg_drawdown_days is None:
            self._avg_drawdown_days = self.drawdown_details['days'].mean()
        return self._avg_drawdown_days

    @property
    def volatility(self):
        if self._volatility is None:
            self._volatility = calc_volatility(self.returns)
        return self._volatility

    @property
    def calmar(self):
        if self._calmar is None:
            self._calmar = self.cagr / abs(self.max_drawdown)
        return self._calmar

    @property
    def skew(self):
        if self._skew is None:
            self._skew = self.returns.skew()
        return self._skew

    @property
    def kurt(self):
        if self._kurt is None:
            self._kurt = self.returns.kurt()
        return self._kurt

    @property
    def mtd_return(self):
        if self._mtd_return is None:
            self._mtd_return = calc_mtd_return(self.returns)
        return self._mtd_return

    @property
    def one_month_return(self):
        if self._one_month_return is None:
            self._one_month_return = calc_1m_return(self.returns)
        return self._one_month_return

    @property
    def three_month_return(self):
        if self._three_month_return is None:
            self._three_month_return = calc_3m_return(self.returns)
        return self._three_month_return

    @property
    def six_month_return(self):
        if self._six_month_return is None:
            self._six_month_return = calc_6m_return(self.returns)
        return self._six_month_return

    @property
    def ytd_return(self):
        if self._ytd_return is None:
            self._ytd_return = calc_ytd_return(self.returns)
        return self._ytd_return

    @property
    def one_year_return(self):
        if self._one_year_return is None:
            self._one_year_return = calc_1y_return(self.returns)
        return self._one_year_return

    @property
    def three_year_return(self):
        if self._three_year_return is None:
            self._three_year_return = calc_3y_return(self.returns)
        return self._three_year_return

    @property
    def best_day(self):
        if self._best_day is None:
            self._best_day = self.returns.max()
        return self._best_day

    @property
    def worst_day(self):
        if self._worst_day is None:
            self._worst_day = self.returns.min()
        return self._worst_day

    @property
    def best_week(self):
        if self._best_week is None:
            self._best_week = self.eow_returns.max()
        return self._best_week

    @property
    def worst_week(self):
        if self._worst_week is None:
            self._worst_week = self.eow_returns.min()
        return self._worst_week

    @property
    def best_month(self):
        if self._best_month is None:
            self._best_month = self.eom_returns.max()
        return self._best_month

    @property
    def worst_month(self):
        if self._worst_month is None:
            self._worst_month = self.eom_returns.min()
        return self._worst_month

    @property
    def win_rate_day(self):
        if self._win_rate_day is None:
            win_count = self.returns[self.returns > 0].count()
            self._win_rate_day = win_count/self.returns.count()
        return self._win_rate_day

    @property
    def win_rate_week(self):
        if self._win_rate_week is None:
            win_count = self.eow_returns[self.eow_returns > 0].count()
            self._win_rate_week = win_count/self.eow_returns.count()
        return self._win_rate_week

    @property
    def win_rate_month(self):
        if self._win_rate_month is None:
            win_count = self.eom_returns[self.eom_returns > 0].count()
            self._win_rate_month = win_count/self.eom_returns.count()
        return self._win_rate_month

    @property
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
        s = pd.Series(dict(_r), name=self._prices.name)
        s.index.name = ''
        return s
