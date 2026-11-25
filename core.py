"""Functions of the notebook Group_G_Permanent_Portfolio.ipynb, copied without modification.

Every function below is an exact copy of its definition in the notebook (Sections 3 to 9, 14, 16 and 17), so
that the Streamlit application computes its results with the same code as the notebook.
tests/test_consistency.py fails if any copy differs from the notebook by a single character.
"""
import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(tickers, start, end):
    # Monthly closing prices adjusted for splits and distributions
    df = yf.download(list(tickers), start=start, end=end,
                     interval='1mo', auto_adjust=True, progress=False)
    prices = df['Close'].copy()
    prices.index = prices.index.tz_localize(None)    # align with the timezone-naive FRED dates
    prices.columns.name = None
    return prices


def download_tbill(start, end):
    # Monthly return on Treasury bills (FRED TB3MS), lagged to the rate known at the start of each month
    last_month = (pd.Timestamp(end) - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    url = ('https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS'
           f'&cosd={start}&coed={last_month}')
    tbill = pd.read_csv(url, index_col=0, parse_dates=True, na_values='.')

    rf = tbill['TB3MS'] / 100 / 12    # percent per annum -> monthly decimal return
    rf = rf.shift(1)
    return rf


def load_data(start, end, tickers=('VTI', 'TLT', 'GLD', 'AGG')):
    # Monthly simple returns of the ETFs and of Treasury bills, validated against the expected calendar
    prices = download_prices(tickers, start, end)
    rf = download_tbill(start, end)

    expected = list(pd.date_range(start, end, freq='MS', inclusive='left').strftime('%Y-%m-%d'))
    if list(prices.index.strftime('%Y-%m-%d')) != expected:
        raise ValueError('Yahoo Finance dates do not match the expected monthly calendar.')
    if list(rf.index.strftime('%Y-%m-%d')) != expected:
        raise ValueError('FRED dates do not match the expected monthly calendar.')

    rets = prices.pct_change(fill_method=None)     # no forward-filling of missing prices
    rets['rf'] = rf.values
    rets = rets.iloc[1:]                           # the first observation has no return

    if rets.isna().any().any():
        bad_months = list(rets.index[rets.isna().any(axis=1)].strftime('%Y-%m'))
        raise ValueError(f'Missing observations in: {bad_months}')
    return prices, rets


def backtest(asset_rets, target_w, band_low, band_high, check_months):
    # Band-rebalanced portfolio with drifting weights.
    # Returns monthly portfolio returns, two-sided turnover and post-trade weights.
    n_months = asset_rets.shape[0]
    values = target_w.copy()                     # position values; initial wealth of $1
    ptf_rets = np.zeros(n_months)
    turnover = np.zeros(n_months)
    weights = np.zeros((n_months, len(target_w)))

    for t in range(n_months):
        old_total = values.sum()
        values = values * (1 + asset_rets.iloc[t].values)
        new_total = values.sum()
        ptf_rets[t] = new_total / old_total - 1
        w = values / new_total                   # pre-trade weights after drift

        # Review at the month-end, using only information available at t
        if asset_rets.index[t].month in check_months and (w.min() < band_low or w.max() > band_high):
            turnover[t] = np.sum(np.abs(target_w - w))
            values = target_w * new_total

        weights[t] = values / values.sum()       # post-trade weights, held over the next month

    return ptf_rets, turnover, weights


def wealth_index(r):
    # Cumulative wealth indexed to 1 in the month preceding the first return
    wealth = (1 + r).cumprod()
    start_date = wealth.index[0] - pd.DateOffset(months=1)
    return pd.concat([pd.Series([1.0], index=[start_date]), wealth])


def max_drawdown(r):
    # Maximum of DD_t = (HWM_t - P_t) / HWM_t, reported as a positive number
    p = wealth_index(r)
    hwm = p.cummax()
    return ((hwm - p) / hwm).max()


def performance(r, rf, bench=None, periods=12):
    # Annualized performance measures from monthly simple returns; beta is not annualized.
    # With a benchmark, adds beta, alpha, the t-statistic of alpha, tracking error and information ratio.
    ex = r - rf
    out = {
        'Annualized average return': r.mean() * periods,
        'Compound annual return': (1 + r).prod() ** (periods / len(r)) - 1,
        'Annualized volatility': r.std() * np.sqrt(periods),
        'Sharpe ratio': ex.mean() / ex.std() * np.sqrt(periods),
        'Beta': np.nan,
        'Annualized alpha': np.nan,
        'Alpha t-statistic': np.nan,
        'Annualized tracking error': np.nan,
        'Information ratio': np.nan,
        'Maximum drawdown': max_drawdown(r),
        'Growth of $1': (1 + r).prod(),
    }
    if bench is not None:
        bench_ex = bench - rf
        beta = ex.cov(bench_ex) / bench_ex.var()
        alpha = ex.mean() - beta * bench_ex.mean()
        resid = ex - alpha - beta * bench_ex
        se_alpha = np.sqrt(resid.var(ddof=2) * (1 / len(r) + bench_ex.mean() ** 2
                                                / ((bench_ex - bench_ex.mean()) ** 2).sum()))
        active = r - bench
        out['Beta'] = beta
        out['Annualized alpha'] = alpha * periods
        out['Alpha t-statistic'] = alpha / se_alpha
        out['Annualized tracking error'] = active.std() * np.sqrt(periods)
        out['Information ratio'] = active.mean() / active.std() * np.sqrt(periods)
    return out


def format_table(table):
    # Presentation only: percentages and ratios to two decimals, '-' where a measure does not apply
    pct_rows = ['Annualized average return', 'Compound annual return', 'Annualized volatility',
                'Annualized alpha', 'Annualized tracking error', 'Maximum drawdown', 'Annual cost drag']
    formatted = {}
    for row in table.index:
        fmt = '{:.2%}' if row in pct_rows else '{:.2f}'
        formatted[row] = ['-' if pd.isna(v) else fmt.format(v) for v in table.loc[row]]
    return pd.DataFrame.from_dict(formatted, orient='index', columns=table.columns)


def months_to_recover(wealth, peak, trough):
    # Months from the equity peak to the first month-end after the trough at which the portfolio
    # has regained its value at the peak; 0 if it stands at or above that value at the trough
    dates = list(wealth.index)
    peak_value = wealth.loc[peak]
    if wealth.loc[trough] >= peak_value:
        return 0
    for i in range(dates.index(trough) + 1, len(dates)):
        if wealth.iloc[i] >= peak_value:
            return i - dates.index(peak)
    return np.nan


def download_cpi(start, end):
    # US consumer price index (FRED CPIAUCNS: CPI-U, all items, not seasonally adjusted) on the monthly calendar
    last_month = (pd.Timestamp(end) - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    url = ('https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS'
           f'&cosd={start}&coed={last_month}')
    cpi = pd.read_csv(url, index_col=0, parse_dates=True, na_values='.')['CPIAUCNS']
    cpi = cpi.reindex(pd.date_range(start, end, freq='MS', inclusive='left'))

    # October 2025 was not published (no price collection during the federal shutdown). As in the BLS's own
    # seasonal adjustment, its level is the geometric mean of September and November.
    gap = pd.Timestamp('2025-10-01')
    before, after = gap - pd.DateOffset(months=1), gap + pd.DateOffset(months=1)
    if before in cpi.index and after in cpi.index and pd.isna(cpi[gap]):
        cpi[gap] = np.sqrt(cpi[before] * cpi[after])

    if cpi.isna().any():
        raise ValueError(f"Missing consumer price index in: {list(cpi.index[cpi.isna()].strftime('%Y-%m'))}")
    return cpi


def real_returns(r, inflation):
    # Returns in real terms, (1 + r) / (1 + inflation) - 1, for every column of r
    return (1 + r).div(1 + inflation, axis=0) - 1


def download_eurusd(start, end):
    # US dollars per euro at each month-end (FRED DEXUSEU: noon buying rates in New York, last business day)
    last_month = (pd.Timestamp(end) - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    url = ('https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU&fq=Monthly&fam=eop'
           f'&cosd={start}&coed={last_month}')
    eurusd = pd.read_csv(url, index_col=0, parse_dates=True, na_values='.')['DEXUSEU']
    eurusd = eurusd.reindex(pd.date_range(start, end, freq='MS', inclusive='left'))
    if eurusd.isna().any():
        raise ValueError(f"Missing exchange rate in: {list(eurusd.index[eurusd.isna()].strftime('%Y-%m'))}")
    return eurusd


def download_euribor(start, end):
    # Monthly return on euro deposits (three-month Euribor from the ECB Data Portal), lagged as the T-bill rate
    last_month = (pd.Timestamp(end) - pd.DateOffset(months=1)).strftime('%Y-%m')
    url = ('https://data-api.ecb.europa.eu/service/data/FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA?format=csvdata'
           f'&startPeriod={pd.Timestamp(start):%Y-%m}&endPeriod={last_month}')
    euribor = pd.read_csv(url, index_col='TIME_PERIOD', parse_dates=True)['OBS_VALUE']
    euribor = euribor.reindex(pd.date_range(start, end, freq='MS', inclusive='left'))
    if euribor.isna().any():
        raise ValueError(f"Missing Euribor rate in: {list(euribor.index[euribor.isna()].strftime('%Y-%m'))}")

    rf_eur = euribor / 100 / 12    # percent per annum (monthly average) -> monthly decimal return
    rf_eur = rf_eur.shift(1)       # each month earns the rate known at its start
    return rf_eur


def download_hicp(start, end):
    # Euro-area consumer prices (FRED CP0000EZCCM086NEST: HICP, all items, changing composition), monthly
    last_month = (pd.Timestamp(end) - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    url = ('https://fred.stlouisfed.org/graph/fredgraph.csv?id=CP0000EZCCM086NEST'
           f'&cosd={start}&coed={last_month}')
    hicp = pd.read_csv(url, index_col=0, parse_dates=True, na_values='.')['CP0000EZCCM086NEST']
    hicp = hicp.reindex(pd.date_range(start, end, freq='MS', inclusive='left'))
    if hicp.isna().any():
        raise ValueError(f"Missing euro-area price index in: {list(hicp.index[hicp.isna()].strftime('%Y-%m'))}")
    return hicp


def in_euros(r, dollar_return):
    # Returns of dollar assets measured in euros, (1 + r) * (1 + dollar_return) - 1, for every column of r
    return (1 + r).mul(1 + dollar_return, axis=0) - 1
