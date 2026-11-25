"""Consistency between the notebook and the application.

1. Every function in core.py is a character-for-character copy of its definition in the notebook.
2. At the notebook's settings, analysis.py reproduces every table the notebook saved in notebook/outputs/,
   starting from the same monthly returns (notebook/outputs/returns_monthly.csv).
3. The engine passes the notebook's Validations 1, 2 and 11.
4. Section 16 (inflation and real returns) is reproduced from the saved returns and consumer price index.
5. Section 17 (the euro-based investor) is reproduced from the saved returns, exchange rates, Euribor and HICP.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analysis  # noqa: E402
import core  # noqa: E402

NOTEBOOK = ROOT / 'notebook' / 'Group_G_Permanent_Portfolio.ipynb'
OUTPUTS = ROOT / 'notebook' / 'outputs'
TOL = 1e-10          # same input data, so results agree up to floating-point rounding


def functions_in(source):
    tree = ast.parse(source)
    return {node.name: ast.get_source_segment(source, node)
            for node in tree.body if isinstance(node, ast.FunctionDef)}


def notebook_functions():
    found = {}
    for cell in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']:
        if cell['cell_type'] == 'code':
            found.update(functions_in(''.join(cell['source'])))
    return found


def saved(name):
    return pd.read_csv(OUTPUTS / name, index_col=0)


@pytest.fixture(scope='module')
def rets():
    return pd.read_csv(OUTPUTS / 'returns_monthly.csv', index_col=0, parse_dates=True)


@pytest.fixture(scope='module')
def cpi():
    return pd.read_csv(OUTPUTS / 'consumer_prices.csv', index_col=0, parse_dates=True)['CPI']


@pytest.fixture(scope='module')
def real(rets, cpi):
    # Section 16: the deflated panel and the simulation repeated on it
    inflation, inflation_12m = analysis.inflation_rates(cpi, rets.index)
    real_rets = core.real_returns(rets, inflation)
    real_results, _, _ = analysis.simulate(real_rets)
    return real_rets, real_results, inflation, inflation_12m


@pytest.fixture(scope='module')
def euro(rets):
    # Section 17: the panel in euros and the simulation repeated on it, nominal and after euro-area inflation
    data = pd.read_csv(OUTPUTS / 'euro_data.csv', index_col=0, parse_dates=True)
    dollar_return = analysis.dollar_returns(data['Dollars per euro'].dropna(), rets.index)
    rf_eur = data.loc[rets.index, 'Euribor return']
    inflation_eur, _ = analysis.inflation_rates(data['HICP'], rets.index)
    euro_rets = core.in_euros(rets, dollar_return)
    euro_results, _, _ = analysis.simulate(euro_rets, rf=rf_eur)
    euro_real_results, _, _ = analysis.simulate(core.real_returns(euro_rets, inflation_eur),
                                                rf=core.real_returns(rf_eur, inflation_eur))
    return {'data': data, 'dollar_return': dollar_return, 'rf_eur': rf_eur, 'euro_rets': euro_rets,
            'results': euro_results, 'real_results': euro_real_results}


EURO_ROW = {'Growth of $1': 'Growth of €1'}       # Section 17 labels the last row of its tables in euros


@pytest.fixture(scope='module')
def model(rets):
    results, weights_after, weights_before = analysis.simulate(rets)
    return results, weights_after, weights_before


def assert_same(ours, theirs):
    pd.testing.assert_frame_equal(ours, theirs, check_dtype=False, check_names=False, check_index_type=False,
                                  check_exact=False, rtol=0, atol=TOL)


# 1. core.py is an exact copy of the notebook's functions

def test_core_functions_are_exact_copies_of_the_notebook():
    nb = notebook_functions()
    ours = functions_in((ROOT / 'core.py').read_text(encoding='utf-8'))
    assert ours, 'core.py defines no functions'
    for name, source in ours.items():
        assert name in nb, f'{name} is not defined in the notebook'
        assert source == nb[name], f'{name} in core.py differs from the notebook'


def test_core_covers_every_function_the_notebook_defines():
    ours = functions_in((ROOT / 'core.py').read_text(encoding='utf-8'))
    assert set(notebook_functions()) == set(ours)


# 2. analysis.py reproduces the notebook's saved tables

def test_parameters_match_section_2():
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    params = next(''.join(c['source']) for c in nb['cells']
                  if c['cell_type'] == 'code' and "start = '" in ''.join(c['source']))
    ns = {'np': np}
    exec(params, ns)
    assert (ns['start'], ns['end']) == (analysis.START, analysis.END)
    assert ns['tickers'] == analysis.TICKERS
    assert np.array_equal(ns['target_w'], analysis.TARGET_W)
    assert (ns['band_low'], ns['band_high']) == (analysis.BAND_LOW, analysis.BAND_HIGH)
    assert ns['check_months'] == analysis.CHECK_MONTHS
    assert ns['w_stocks_bench'] == analysis.W_STOCKS_BENCH
    assert (ns['cost'], ns['cost_stress']) == (analysis.COST, analysis.COST_STRESS)
    assert ns['periods'] == analysis.PERIODS


def test_monthly_results(model):
    results, _, _ = model
    theirs = pd.read_csv(OUTPUTS / 'results_monthly.csv', index_col=0, parse_dates=True)
    assert_same(results, theirs)


def test_table1_performance(model):
    perf_table, perf_vs_stocks = analysis.performance_tables(model[0])
    assert_same(perf_table, saved('table1_performance.csv'))
    assert_same(perf_vs_stocks, saved('table1_relative_to_all_equity.csv'))


def test_table1_formatted(model):
    perf_table, _ = analysis.performance_tables(model[0])
    theirs = saved('table1_performance_formatted.csv').fillna('-')
    assert (core.format_table(perf_table) == theirs).all().all()


def test_table2_costs_and_sensitivity(model):
    results, _, weights_before = model
    cost_summary, sensitivity = analysis.cost_analysis(results)
    theirs = saved('table2_cost_summary.csv')['Value']
    assert np.allclose(cost_summary.values, theirs.values, rtol=0, atol=TOL)
    assert list(cost_summary.index) == list(theirs.index)
    assert_same(sensitivity, saved('cost_sensitivity.csv'))
    assert_same(analysis.rebalancing_record(results, weights_before), saved('table2_rebalancing_record.csv'))


def test_crisis_tables(model):
    crisis_table, recovery_table = analysis.crisis_analysis(model[0])
    assert_same(crisis_table, saved('crisis_returns.csv'))
    assert_same(recovery_table, saved('crisis_recovery_months.csv'))


def test_rebalancing_rules(rets, model):
    assert_same(analysis.rule_comparison(rets, model[0]), saved('rebalancing_rules.csv'))


def test_calendar_returns(rets):
    ours = analysis.calendar_returns(rets)
    theirs = saved('calendar_returns.csv')
    assert_same(ours, theirs)


# 3. Validations 1 and 2 of the notebook

def test_validation_1_buy_and_hold(rets):
    assets = rets[['VTI', 'TLT', 'GLD', 'rf']]
    r, turnover, _ = core.backtest(assets, analysis.TARGET_W, 0.15, 0.35, check_months=[])
    buy_and_hold = np.sum(analysis.TARGET_W * (1 + assets).prod().values)
    assert abs(np.prod(1 + r) - buy_and_hold) < 1e-9
    assert turnover.sum() == 0


def test_validation_2_constant_weights(rets):
    assets = rets[['VTI', 'TLT', 'GLD', 'rf']]
    r, _, _ = core.backtest(assets, analysis.TARGET_W, 0.25, 0.25, check_months=list(range(1, 13)))
    formula = 0.25 * (rets['VTI'] + rets['TLT'] + rets['GLD'] + rets['rf'])
    assert np.max(np.abs(r - formula.values)) < 1e-9


def test_wealth_index_starts_at_one(model):
    assert core.wealth_index(model[0]['pp_net']).iloc[0] == 1.0


def test_validation_11_deflation_leaves_rebalancing_unchanged(model, real):
    results = model[0]
    real_results, inflation = real[1], real[2]
    assert np.max(np.abs(real_results['turnover'] - results['turnover'])) < 1e-12
    deflated = (1 + results['pp']) / (1 + inflation) - 1
    assert np.max(np.abs(real_results['pp'] - deflated)) < 1e-12


# 4. Section 16: inflation and real returns

def test_section_16_parameters_match_the_notebook():
    sources = [''.join(c['source']) for c in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
               if c['cell_type'] == 'code']
    assert any(f"cpi = download_cpi('{analysis.CPI_START}', end)" in s for s in sources)
    assert any(f'regime_bounds = {analysis.INFLATION_BOUNDS}' in s for s in sources)


def test_saved_inflation_rates(cpi, real):
    saved_cpi = pd.read_csv(OUTPUTS / 'consumer_prices.csv', index_col=0, parse_dates=True)
    assert saved_cpi.index[0] == pd.Timestamp(analysis.CPI_START) and not saved_cpi['CPI'].isna().any()
    inflation, inflation_12m = real[2], real[3]
    assert np.allclose(inflation, saved_cpi.loc[inflation.index, 'Monthly inflation'], rtol=0, atol=TOL)
    assert np.allclose(inflation_12m, saved_cpi.loc[inflation.index, '12-month inflation'], rtol=0, atol=TOL)


def test_real_returns_panel(real):
    theirs = pd.read_csv(OUTPUTS / 'real_returns_monthly.csv', index_col=0, parse_dates=True)
    assert_same(real[0], theirs)


def test_real_results_monthly(real):
    theirs = pd.read_csv(OUTPUTS / 'real_results_monthly.csv', index_col=0, parse_dates=True)
    assert_same(real[1], theirs)


def test_table1_real_terms(real):
    table, _ = analysis.performance_tables(real[1])
    assert_same(table, saved('table1_real_terms.csv'))
    theirs = saved('table1_real_terms_formatted.csv').fillna('-')
    assert (core.format_table(table) == theirs).all().all()


def test_inflation_regimes(real):
    table, regime = analysis.inflation_regimes(real[0], real[1], real[3])
    assert_same(table, saved('inflation_regimes.csv'))
    assert int(table.loc['Months'].sum()) == len(regime) == len(real[0])


def test_calendar_real_returns(real):
    assert_same(analysis.calendar_real_returns(real[0], real[1], real[2]), saved('calendar_real_returns.csv'))


def test_download_cpi_fills_only_october_2025(monkeypatch):
    # FRED's response replaced by a series with October 2025 missing, as published (no network access)
    months = pd.date_range('2025-07-01', '2026-01-01', freq='MS', inclusive='left')
    fred = pd.DataFrame({'CPIAUCNS': [100.0, 101.0, 102.0, np.nan, 104.04, 105.0]}, index=months)
    monkeypatch.setattr(core.pd, 'read_csv', lambda *args, **kwargs: fred.copy())
    cpi = core.download_cpi('2025-07-01', '2026-01-01')
    assert cpi[pd.Timestamp('2025-10-01')] == pytest.approx(np.sqrt(102.0 * 104.04))
    assert (cpi.drop(pd.Timestamp('2025-10-01')).values == fred['CPIAUCNS'].dropna().values).all()

    # Any other missing month stops the analysis
    gappy = fred.copy()
    gappy.iloc[1, 0] = np.nan
    monkeypatch.setattr(core.pd, 'read_csv', lambda *args, **kwargs: gappy.copy())
    with pytest.raises(ValueError, match='2025-08'):
        core.download_cpi('2025-07-01', '2026-01-01')


# 5. Section 17: the perspective of a euro-based investor

def test_section_17_parameters_match_the_notebook():
    sources = [''.join(c['source']) for c in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
               if c['cell_type'] == 'code']
    assert any('eurusd = download_eurusd(start, end)' in s for s in sources)
    assert any('rf_eur = download_euribor(start, end).loc[rets.index]' in s for s in sources)
    assert any(f"hicp = download_hicp('{analysis.HICP_START}', end)" in s for s in sources)


def test_saved_euro_data(rets, euro):
    data = euro['data']
    assert data.index[0] == pd.Timestamp(analysis.HICP_START) and not data['HICP'].isna().any()
    assert data['Dollars per euro'].first_valid_index() == pd.Timestamp(analysis.START)
    assert not data.loc[analysis.START:, 'Dollars per euro'].isna().any()
    assert not euro['rf_eur'].isna().any()
    assert np.allclose(euro['dollar_return'], data.loc[rets.index, 'Dollar return in euros'], rtol=0, atol=TOL)
    inflation_eur, inflation_eur_12m = analysis.inflation_rates(data['HICP'], rets.index)
    assert np.allclose(inflation_eur, data.loc[rets.index, 'Monthly euro-area inflation'], rtol=0, atol=TOL)
    assert np.allclose(inflation_eur_12m, data.loc[rets.index, '12-month euro-area inflation'], rtol=0, atol=TOL)


def test_validation_12_conversion_leaves_rebalancing_unchanged(model, euro):
    results = model[0]
    for run in (euro['results'], euro['real_results']):
        assert np.max(np.abs(run['turnover'] - results['turnover'])) < 1e-12
    converted = (1 + results['pp']) * (1 + euro['dollar_return']) - 1
    assert np.max(np.abs(euro['results']['pp'] - converted)) < 1e-12


def test_euro_returns_panel(euro):
    theirs = pd.read_csv(OUTPUTS / 'euro_returns_monthly.csv', index_col=0, parse_dates=True)
    assert_same(euro['euro_rets'], theirs)


def test_euro_results_monthly(euro):
    for key, name in (('results', 'euro_results_monthly.csv'), ('real_results', 'euro_real_results_monthly.csv')):
        theirs = pd.read_csv(OUTPUTS / name, index_col=0, parse_dates=True)
        assert_same(euro[key], theirs)


def test_table1_in_euros(euro):
    for key, name in (('results', 'table1_euros'), ('real_results', 'table1_euros_real')):
        table, _ = analysis.performance_tables(euro[key])
        table = table.rename(index=EURO_ROW)
        assert_same(table, saved(f'{name}.csv'))
        theirs = saved(f'{name}_formatted.csv').fillna('-')
        assert (core.format_table(table) == theirs).all().all()


def test_perspective_table(model, real, euro):
    tables = {'Dollars': analysis.performance_tables(model[0])[0],
              'Dollars, real': analysis.performance_tables(real[1])[0],
              'Euros': analysis.performance_tables(euro['results'])[0].rename(index=EURO_ROW),
              'Euros, real': analysis.performance_tables(euro['real_results'])[0].rename(index=EURO_ROW)}
    theirs = pd.read_csv(OUTPUTS / 'perspectives.csv', index_col=0, header=[0, 1])
    assert_same(analysis.perspective_table(tables), theirs)


def test_currency_effects(rets, model, euro):
    ours = analysis.currency_effects(rets, model[0], euro['euro_rets'], euro['results'], euro['dollar_return'])
    assert_same(ours, saved('currency_effects.csv'))


def test_crises_in_euros(model, euro):
    crisis_table, recovery_table = analysis.crisis_analysis(model[0])
    ours, ours_recovery = analysis.crises_on_dates(euro['results'], crisis_table,
                                                   dollar_return=euro['dollar_return'])
    assert_same(ours, saved('crisis_returns_euros.csv'))
    assert_same(ours_recovery, saved('crisis_recovery_months_euros.csv'))
    # In dollars, the same calculation reproduces Section 14
    same, same_recovery = analysis.crises_on_dates(model[0], crisis_table)
    assert_same(same, crisis_table)
    assert_same(same_recovery, recovery_table)


def test_calendar_euro_returns(euro):
    ours = analysis.calendar_euro_returns(euro['euro_rets'], euro['results'], euro['dollar_return'], euro['rf_eur'])
    assert_same(ours, saved('calendar_euro_returns.csv'))


def test_in_euros_multiplies_by_the_change_in_the_dollar():
    r = pd.DataFrame({'a': [0.10, -0.05]})
    e = pd.Series([0.02, -0.01])
    expected = pd.DataFrame({'a': [1.10 * 1.02 - 1, 0.95 * 0.99 - 1]})
    pd.testing.assert_frame_equal(core.in_euros(r, e), expected)


def test_download_eurusd_requires_every_month(monkeypatch):
    # FRED's response replaced by a month-end series (no network access)
    months = pd.date_range('2025-01-01', '2025-05-01', freq='MS', inclusive='left')
    fred = pd.DataFrame({'DEXUSEU': [1.04, 1.04, 1.08, 1.13]}, index=months)
    monkeypatch.setattr(core.pd, 'read_csv', lambda *args, **kwargs: fred.copy())
    assert list(core.download_eurusd('2025-01-01', '2025-05-01')) == [1.04, 1.04, 1.08, 1.13]
    gappy = fred.copy()
    gappy.iloc[2, 0] = np.nan
    monkeypatch.setattr(core.pd, 'read_csv', lambda *args, **kwargs: gappy.copy())
    with pytest.raises(ValueError, match='2025-03'):
        core.download_eurusd('2025-01-01', '2025-05-01')


def test_download_euribor_converts_and_lags(monkeypatch):
    # The ECB's response replaced by three monthly averages in percent per annum
    months = pd.date_range('2025-01-01', '2025-04-01', freq='MS', inclusive='left')
    ecb = pd.DataFrame({'OBS_VALUE': [2.4, 2.4, 1.2]}, index=pd.Index(months, name='TIME_PERIOD'))
    monkeypatch.setattr(core.pd, 'read_csv', lambda *args, **kwargs: ecb.copy())
    rf_eur = core.download_euribor('2025-01-01', '2025-04-01')
    assert np.isnan(rf_eur.iloc[0])
    assert rf_eur.iloc[1:].tolist() == pytest.approx([0.002, 0.002])     # the previous month's rate / 100 / 12
    monkeypatch.setattr(core.pd, 'read_csv', lambda *args, **kwargs: ecb.iloc[:2].copy())
    with pytest.raises(ValueError, match='2025-03'):
        core.download_euribor('2025-01-01', '2025-04-01')


def test_download_hicp_requires_every_month(monkeypatch):
    months = pd.date_range('2025-01-01', '2025-04-01', freq='MS', inclusive='left')
    fred = pd.DataFrame({'CP0000EZCCM086NEST': [100.0, np.nan, 100.6]}, index=months)
    monkeypatch.setattr(core.pd, 'read_csv', lambda *args, **kwargs: fred.copy())
    with pytest.raises(ValueError, match='2025-02'):
        core.download_hicp('2025-01-01', '2025-04-01')
