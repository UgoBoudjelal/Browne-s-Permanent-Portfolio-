"""Calculations of the notebook's result cells, written as functions of the model parameters.

core.py holds exact copies of the notebook's functions. This module repeats the code of the cells that
use them (Sections 7, 8, 10 and 13 to 17), with the parameters of Section 2 as arguments, so that the
application can re-run the analysis under the assumptions chosen by the user. In real terms, the application
passes the return panel deflated by core.real_returns to the same functions, as Section 16 does; in euros, the
panel converted by core.in_euros and the Euribor rate as the risk-free rate, as Section 17 does.
tests/test_consistency.py checks that, at the notebook's settings, every function reproduces the tables the
notebook saved in notebook/outputs/.
"""
import numpy as np
import pandas as pd

from core import backtest, months_to_recover, performance, wealth_index

# Model parameters of the notebook (Section 2)
START, END = '2004-12-01', '2026-11-01'
TICKERS = ['VTI', 'TLT', 'GLD', 'AGG']
TARGET_W = np.array([0.25, 0.25, 0.25, 0.25])
BAND_LOW, BAND_HIGH = 0.15, 0.35
CHECK_MONTHS = list(range(1, 13))
W_STOCKS_BENCH = 0.60
COST, COST_STRESS = 0.0005, 0.005
PERIODS = 12
CPI_START = '2004-01-01'            # Section 16: the price index from twelve months before the first return
INFLATION_BOUNDS = (0.01, 0.03)     # Section 16: thresholds of the low, moderate and high inflation regimes
HICP_START = '2004-01-01'           # Section 17: the euro-area price index from the same month as the CPI

SLEEVES = ['Equities', 'Long-term Treasuries', 'Gold', 'Treasury bills']
REVIEW_SCHEDULES = {'Monthly': list(range(1, 13)), 'Quarterly': [3, 6, 9, 12],
                    'Semi-annual': [6, 12], 'Annual': [12]}
COST_LEVELS = [0, 0.0005, 0.001, 0.0025, 0.005, 0.01]
CRISIS_RANGES = {
    'Financial crisis (2007-09)': ('2007-01', '2009-12'),
    'Debt crises (2011)': ('2011-01', '2011-12'),
    'Q4 2018 sell-off': ('2018-07', '2019-03'),
    'COVID-19 crash (2020)': ('2019-10', '2020-12'),
    'Rate shock (2022)': ('2021-10', '2022-12'),
}
MEASURES_SHOWN = ['Annualized average return', 'Compound annual return', 'Annualized volatility', 'Sharpe ratio',
                  'Maximum drawdown', 'Annualized alpha', 'Alpha t-statistic', 'Information ratio']


def bench_name(w_stocks_bench):
    # Label of the stock-bond benchmark, e.g. '60/40'
    return f'{round(100 * w_stocks_bench)}/{round(100 * (1 - w_stocks_bench))}'


def simulate(rets, band_low=BAND_LOW, band_high=BAND_HIGH, check_months=CHECK_MONTHS,
             cost=COST, cost_stress=COST_STRESS, w_stocks_bench=W_STOCKS_BENCH, rf=None):
    # Sections 7 and 8: strategy before and after costs, benchmark before and after costs. The risk-free rate
    # of the performance measures is the T-bill return of the panel, or rf if given (Euribor in Section 17).
    pp_assets = rets[['VTI', 'TLT', 'GLD', 'rf']]          # ordered as TARGET_W
    pp_rets, pp_turnover, pp_weights = backtest(pp_assets, TARGET_W, band_low, band_high, check_months)

    results = pd.DataFrame(index=rets.index)
    results['pp'] = pp_rets
    results['bench_6040'] = w_stocks_bench * rets['VTI'] + (1 - w_stocks_bench) * rets['AGG']
    results['stocks'] = rets['VTI']
    results['rf'] = rets['rf'] if rf is None else rf
    results['turnover'] = pp_turnover
    results['pp_net'] = results['pp'] - results['turnover'] * cost
    results['pp_net_stress'] = results['pp'] - results['turnover'] * cost_stress
    w_stocks_drifted = w_stocks_bench * (1 + rets['VTI']) / (1 + results['bench_6040'])
    results['bench_turnover'] = 2 * (w_stocks_drifted - w_stocks_bench).abs()
    results['bench_6040_net'] = results['bench_6040'] - results['bench_turnover'] * cost

    # Section 7, Validation 3: post-trade weights and the pre-trade weights reconstructed from them
    weights_after = pd.DataFrame(pp_weights, index=rets.index, columns=SLEEVES)
    weights_before = weights_after.shift(1).fillna(0.25) * (1 + pp_assets.values)
    weights_before = weights_before.div(weights_before.sum(axis=1), axis=0)
    return results, weights_after, weights_before


def performance_tables(results, bench='60/40'):
    # Section 10: Table 1 and the measures relative to the all-equity portfolio
    perf_table = pd.DataFrame({
        'Permanent Portfolio': performance(results['pp_net'], results['rf'], results['bench_6040']),
        'PP, stress costs': performance(results['pp_net_stress'], results['rf'], results['bench_6040']),
        bench: performance(results['bench_6040'], results['rf']),
        f'{bench}, net of costs': performance(results['bench_6040_net'], results['rf']),
        'All equity (VTI)': performance(results['stocks'], results['rf']),
    })
    perf_vs_stocks = pd.DataFrame({
        'Permanent Portfolio vs all equity': performance(results['pp_net'], results['rf'], results['stocks'])
    }).loc[['Beta', 'Annualized alpha', 'Alpha t-statistic', 'Annualized tracking error', 'Information ratio']]
    return perf_table, perf_vs_stocks


def cost_analysis(results, cost=COST, cost_stress=COST_STRESS, cost_levels=COST_LEVELS):
    # Sections 8 and 13: turnover, cost drag, sensitivity to the cost rate and break-even cost
    n_rebalances = int((results['turnover'] > 0).sum())
    n_years = len(results) / PERIODS
    annual_turnover = results['turnover'].sum() / n_years
    bench_annual_turnover = results['bench_turnover'].sum() / n_years

    sensitivity = pd.DataFrame()
    for c in cost_levels:
        net = results['pp'] - results['turnover'] * c
        m = performance(net, results['rf'], results['bench_6040'])
        m['Annual cost drag'] = annual_turnover * c
        label = f'{c:.2%}'
        if abs(c - cost) < 1e-12:
            label += ' (base case)'
        if abs(c - cost_stress) < 1e-12:
            label += ' (stress case)'
        sensitivity[label] = pd.Series(m)

    alpha_no_cost = performance(results['pp'], results['rf'], results['bench_6040'])['Annualized alpha']
    alpha_full_cost = performance(results['pp'] - results['turnover'] * 1.0,
                                  results['rf'], results['bench_6040'])['Annualized alpha']
    breakeven = alpha_no_cost / (alpha_no_cost - alpha_full_cost) if alpha_no_cost > 0 else np.nan

    cost_summary = pd.Series({
        'Rebalancing events': n_rebalances,
        'Years in sample': n_years,
        'Average annual turnover': annual_turnover,
        'Average annual cost, base case': annual_turnover * cost,
        'Average annual cost, stress case': annual_turnover * cost_stress,
        '60/40 average annual turnover': bench_annual_turnover,
        '60/40 average annual cost, base case': bench_annual_turnover * cost,
        '60/40 average annual cost, stress case': bench_annual_turnover * cost_stress,
        'Break-even cost rate': breakeven,
    }, name='Value')
    return cost_summary, sensitivity


def rebalancing_record(results, weights_before, cost=COST, cost_stress=COST_STRESS):
    # Appendix: pre-trade weights, turnover and cost of each rebalancing
    traded = results['turnover'] > 0
    record = weights_before[traded].copy()
    record['Turnover'] = results.loc[traded, 'turnover']
    record['Cost, base case'] = record['Turnover'] * cost
    record['Cost, stress case'] = record['Turnover'] * cost_stress
    record.index = record.index.strftime('%Y-%m')
    return record


def crisis_analysis(results, bench='60/40', crisis_ranges=CRISIS_RANGES):
    # Section 14: return from equity peak to equity trough and months to recovery
    portfolios = {'Permanent Portfolio': wealth_index(results['pp_net']),
                  bench: wealth_index(results['bench_6040']),
                  'All equity (VTI)': wealth_index(results['stocks'])}
    stock_wealth = portfolios['All equity (VTI)']
    dates = list(stock_wealth.index)

    crisis_rows = []
    recovery_rows = []
    for name in crisis_ranges:
        first, last = crisis_ranges[name]
        # Application only: skip episodes whose search window is not entirely inside the chosen period
        if pd.Timestamp(first) < dates[0] or pd.Timestamp(last) > dates[-1]:
            continue
        window = stock_wealth.loc[first:last]
        trough = window.idxmin()
        peak = window.loc[:trough].idxmax()
        if peak == trough:
            continue
        row = {'Episode': name, 'Equity peak': f'{peak:%b %Y}', 'Equity trough': f'{trough:%b %Y}',
               'Months, peak to trough': dates.index(trough) - dates.index(peak)}
        recovery = {'Episode': name}
        for p in portfolios:
            row[p] = portfolios[p].loc[trough] / portfolios[p].loc[peak] - 1
            recovery[p] = months_to_recover(portfolios[p], peak, trough)
        crisis_rows.append(row)
        recovery_rows.append(recovery)

    columns = ['Episode', 'Equity peak', 'Equity trough', 'Months, peak to trough'] + list(portfolios)
    crisis_table = pd.DataFrame(crisis_rows, columns=columns).set_index('Episode')
    recovery_table = pd.DataFrame(recovery_rows, columns=['Episode'] + list(portfolios)).set_index('Episode')
    return crisis_table, recovery_table


def rule_comparison(rets, results, cost=COST, band_low=BAND_LOW, band_high=BAND_HIGH, check_months=CHECK_MONTHS):
    # Section 15: the strategy under alternative rebalancing rules, net of costs, relative to the benchmark
    pp_assets = rets[['VTI', 'TLT', 'GLD', 'rf']]
    n_years = len(results) / PERIODS
    review_schedules = {'monthly': list(range(1, 13)), 'quarterly': [3, 6, 9, 12], 'annual': [12]}
    band_choices = {'20-30%': (0.20, 0.30), '15-35%': (0.15, 0.35), '10-40%': (0.10, 0.40)}

    rules = {}
    for band_label in band_choices:
        for review_label in review_schedules:
            low, high = band_choices[band_label]
            name = f'Band {band_label}, {review_label} review'
            if (low, high, review_schedules[review_label]) == (band_low, band_high, check_months):
                name += ' (strategy)'
            rules[name] = (low, high, review_schedules[review_label])
    for review_label in review_schedules:
        rules[f'Calendar reset, {review_label}'] = (0.25, 0.25, review_schedules[review_label])
    rules['Buy and hold'] = (0.25, 0.25, [])

    rule_rows = []
    for name in rules:
        low, high, months = rules[name]
        r, to, _ = backtest(pp_assets, TARGET_W, low, high, months)
        net = pd.Series(r - to * cost, index=rets.index)
        m = performance(net, results['rf'], results['bench_6040'])
        row = {'Rule': name, 'Rebalances': int((to > 0).sum()), 'Annual turnover': to.sum() / n_years}
        for k in MEASURES_SHOWN:
            row[k] = m[k]
        rule_rows.append(row)
    return pd.DataFrame(rule_rows).set_index('Rule')


def calendar_returns(df):
    # Section 7: compounded return of each calendar year (a partial year covers the months in the sample)
    return (1 + df).groupby(df.index.year).prod() - 1


def inflation_rates(cpi, index):
    # Section 16: monthly inflation and inflation over the preceding twelve months, on the months of the panel
    monthly = cpi.pct_change(fill_method=None).loc[index]
    annual = cpi.pct_change(12, fill_method=None).loc[index]
    return monthly, annual


def inflation_regimes(real_rets, real_results, inflation_12m, bounds=INFLATION_BOUNDS, bench='60/40'):
    # Section 16: annualized compound real return of each sleeve and portfolio in the months of low, moderate
    # and high inflation over the preceding twelve months; also returns the regime of each month
    bound_low, bound_high = bounds
    labels = [f'Low ({100 * bound_low:g}% or less)', f'Moderate ({100 * bound_low:g}% to {100 * bound_high:g}%)',
              f'High (above {100 * bound_high:g}%)']
    regime = pd.cut(inflation_12m, [-np.inf, bound_low, bound_high, np.inf], labels=labels)

    series = {'Equities (VTI)': real_rets['VTI'], 'Long-term Treasuries (TLT)': real_rets['TLT'],
              'Gold (GLD)': real_rets['GLD'], 'Treasury bills': real_rets['rf'],
              'Permanent Portfolio': real_results['pp_net'], bench: real_results['bench_6040'],
              'Aggregate bonds (AGG)': real_rets['AGG']}
    table = pd.DataFrame(index=labels)
    for name in series:
        r = series[name]
        table[name] = [(1 + r[regime == k]).prod() ** (PERIODS / (regime == k).sum()) - 1
                       if (regime == k).any() else np.nan for k in labels]
    table['Months'] = [int((regime == k).sum()) for k in labels]
    table['Average 12-month inflation'] = [inflation_12m[regime == k].mean() for k in labels]
    return table.T, regime


def calendar_real_returns(real_rets, real_results, inflation, bench='60/40'):
    # Section 16: calendar-year inflation (December to December) and real returns
    return calendar_returns(pd.DataFrame({
        'Inflation': inflation,
        'Permanent Portfolio': real_results['pp_net'],
        bench: real_results['bench_6040'],
        'All equity (VTI)': real_rets['VTI'],
        'Long-term Treasuries (TLT)': real_rets['TLT'],
        'Gold (GLD)': real_rets['GLD'],
        'Treasury bills': real_rets['rf'],
    }))


def dollar_returns(eurusd, index):
    # Section 17: return in euros of holding one dollar, from the dollars per euro at each month-end
    return (eurusd.shift(1) / eurusd - 1).loc[index]


def perspective_table(tables, bench='60/40'):
    # Section 17: the main measures of the strategy and the benchmark in each currency, nominal and real;
    # tables maps 'Dollars', 'Dollars, real', 'Euros' and 'Euros, real' to the corresponding Table 1
    summary_rows = ['Compound annual return', 'Annualized volatility', 'Sharpe ratio', 'Maximum drawdown',
                    'Annualized alpha', 'Alpha t-statistic']
    return pd.concat({p: pd.DataFrame({k: tables[k][p] for k in tables}).loc[summary_rows]
                      for p in ['Permanent Portfolio', bench]}, axis=1)


def currency_effects(rets, results, euro_rets, euro_results, dollar_return, bench='60/40'):
    # Section 17: each series in dollars and in euros, and the correlation of its dollar return with the
    # dollar's return in euros
    usd_series = pd.DataFrame({'Equities (VTI)': rets['VTI'], 'Long-term Treasuries (TLT)': rets['TLT'],
                               'Gold (GLD)': rets['GLD'], 'Treasury bills': rets['rf'],
                               'Aggregate bonds (AGG)': rets['AGG'], 'Permanent Portfolio': results['pp_net'],
                               bench: results['bench_6040']})
    eur_series = pd.DataFrame({'Equities (VTI)': euro_rets['VTI'], 'Long-term Treasuries (TLT)': euro_rets['TLT'],
                               'Gold (GLD)': euro_rets['GLD'], 'Treasury bills': euro_rets['rf'],
                               'Aggregate bonds (AGG)': euro_rets['AGG'],
                               'Permanent Portfolio': euro_results['pp_net'], bench: euro_results['bench_6040']})
    return pd.DataFrame({
        'Compound annual return in dollars': (1 + usd_series).prod() ** (PERIODS / len(rets)) - 1,
        'Compound annual return in euros': (1 + eur_series).prod() ** (PERIODS / len(rets)) - 1,
        'Volatility in dollars': usd_series.std() * np.sqrt(PERIODS),
        'Volatility in euros': eur_series.std() * np.sqrt(PERIODS),
        'Correlation with the dollar': usd_series.corrwith(dollar_return),
    })


def crises_on_dates(results, crisis_table, bench='60/40', dollar_return=None):
    # Section 17: the episodes of Section 14, with the same peak and trough dates, measured with other returns
    # (in euros, or after inflation in the application); with dollar_return, also the dollar against the euro
    wealth = pd.DataFrame({'Permanent Portfolio': wealth_index(results['pp_net']),
                           bench: wealth_index(results['bench_6040']),
                           'All equity (VTI)': wealth_index(results['stocks'])})
    peaks = pd.to_datetime(crisis_table['Equity peak'], format='%b %Y')
    troughs = pd.to_datetime(crisis_table['Equity trough'], format='%b %Y')

    table = crisis_table[['Equity peak', 'Equity trough', 'Months, peak to trough']].copy()
    recovery_table = pd.DataFrame(index=crisis_table.index)
    for p in wealth:
        w = wealth[p]
        table[p] = [w.loc[t] / w.loc[k] - 1 for k, t in zip(peaks, troughs)]
        recovery_table[p] = [months_to_recover(w, k, t) for k, t in zip(peaks, troughs)]
    if dollar_return is not None:
        dollar_value = wealth_index(dollar_return)
        table['Dollar against the euro'] = [dollar_value.loc[t] / dollar_value.loc[k] - 1
                                            for k, t in zip(peaks, troughs)]
    return table, recovery_table


def calendar_euro_returns(euro_rets, euro_results, dollar_return, rf_eur, bench='60/40'):
    # Section 17: calendar-year returns in euros and the return of the dollar against the euro
    return calendar_returns(pd.DataFrame({
        'Dollar against the euro': dollar_return,
        'Permanent Portfolio': euro_results['pp_net'],
        bench: euro_results['bench_6040'],
        'All equity (VTI)': euro_rets['VTI'],
        'Long-term Treasuries (TLT)': euro_rets['TLT'],
        'Gold (GLD)': euro_rets['GLD'],
        'Treasury bills': euro_rets['rf'],
        'Euribor': rf_eur,
    }))
