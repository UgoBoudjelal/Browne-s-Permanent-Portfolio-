"""Consistency between the notebook and the application.

1. Every function in core.py is a character-for-character copy of its definition in the notebook.
2. At the notebook's settings, analysis.py reproduces every table the notebook saved in notebook/outputs/,
   starting from the same monthly returns (notebook/outputs/returns_monthly.csv).
3. The engine passes the notebook's Validations 1 and 2.
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
