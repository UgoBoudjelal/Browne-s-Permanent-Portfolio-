"""The Streamlit application, run headless with the notebook's saved returns (no network access)."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / 'app.py')
OUTPUTS = ROOT / 'notebook' / 'outputs'


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('PP_APP_OFFLINE', '1')
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.fixture
def app(client):
    # The analyst view renders every section of the app
    client.segmented_control(key='view').set_value('Analyst').run()
    assert not client.exception, client.exception
    return client


def performance_table(at):
    # Table 1 is the table with the stress-cost column
    return next(t.value for t in at.table if 'PP, stress costs' in t.value.columns)


def test_client_view_opens_on_the_overview(client):
    assert client.segmented_control(key='view').value == 'Client'
    assert any('In brief' in m.value for m in client.markdown)
    assert any('reproduces Table 1' in s.value for s in client.success)     # Method and data is in both views


def test_default_settings_reproduce_table1(app):
    shown = performance_table(app).replace('–', '-')      # the app shows "not applicable" as an en dash
    expected = pd.read_csv(OUTPUTS / 'table1_performance_formatted.csv', index_col=0).fillna('-')
    assert list(shown.columns) == list(expected.columns)
    assert (shown.values == expected.values).all()
    assert any('reproduces Table 1' in s.value for s in app.success)


def test_kpis_match_the_notebook(app):
    table = pd.read_csv(OUTPUTS / 'table1_performance.csv', index_col=0)
    sharpe = next(m for m in app.metric if m.label == 'Sharpe ratio')
    assert sharpe.value == f"{table.loc['Sharpe ratio', 'Permanent Portfolio']:.2f}"


@pytest.mark.parametrize('rule', ['Tolerance band', 'Calendar reset', 'Buy and hold'])
@pytest.mark.parametrize('review', ['Monthly', 'Quarterly', 'Semi-annual', 'Annual'])
def test_every_rule_runs(app, rule, review):
    app.segmented_control(key='rule').set_value(rule)
    app.segmented_control(key='review').set_value(review)
    app.run()
    assert not app.exception


def test_changing_the_cost_changes_the_results(app):
    before = performance_table(app).loc['Annualized average return', 'Permanent Portfolio']
    app.slider(key='cost').set_value(1.0).run()
    assert not app.exception
    after = performance_table(app).loc['Annualized average return', 'Permanent Portfolio']
    assert before != after


def test_benchmark_and_period_inputs(app):
    app.slider(key='bench_equity').set_value(100).run()
    assert not app.exception
    assert '100/0' in performance_table(app).columns
    app.select_slider(key='period').set_value(('2016-01', '2026-10')).run()
    assert not app.exception


def test_short_period_is_refused(app):
    app.select_slider(key='period').set_value(('2020-01', '2021-06')).run()
    assert not app.exception
    assert app.warning


def test_reset_restores_the_notebook_settings(app):
    app.slider(key='cost').set_value(0.5).run()
    app.button(key='reset').click().run()
    assert not app.exception
    assert app.slider(key='cost').value == 0.05
    assert any('reproduces Table 1' in s.value for s in app.success)


# Real terms (Section 16 of the notebook)

def test_real_view_reproduces_the_real_terms_table(app):
    app.segmented_control(key='terms').set_value('Real').run()
    assert not app.exception
    shown = performance_table(app).replace('–', '-')
    expected = pd.read_csv(OUTPUTS / 'table1_real_terms_formatted.csv', index_col=0).fillna('-')
    assert (shown.values == expected.values).all()
    table = pd.read_csv(OUTPUTS / 'table1_real_terms.csv', index_col=0)
    sharpe = next(m for m in app.metric if m.label == 'Sharpe ratio')
    assert sharpe.value == f"{table.loc['Sharpe ratio', 'Permanent Portfolio']:.2f}"
    assert any(m.label.startswith('Real value of') for m in app.metric)


def test_method_checks_both_tables(app):
    messages = [s.value for s in app.success]
    assert any('reproduces Table 1' in m for m in messages)
    assert any('real-terms Table 1 of Section 16' in m for m in messages)


def test_inflation_tab_summarizes_the_regimes(app):
    briefs = [m.value for m in app.markdown if 'US consumer prices rose by' in m.value]
    assert briefs
    regimes = pd.read_csv(OUTPUTS / 'inflation_regimes.csv', index_col=0)
    high = regimes.loc['Permanent Portfolio', 'High (above 3%)']
    assert any(f'{high:.1%}' in b for b in briefs)


@pytest.mark.parametrize('rule', ['Tolerance band', 'Calendar reset', 'Buy and hold'])
def test_every_rule_runs_in_real_terms(app, rule):
    app.segmented_control(key='terms').set_value('Real')
    app.segmented_control(key='rule').set_value(rule)
    app.run()
    assert not app.exception


def test_inflation_thresholds_and_a_period_without_low_inflation(app):
    # Twelve-month inflation never fell to 1% between 2021 and 2026, so the low regime is empty
    app.select_slider(key='period').set_value(('2021-01', '2026-10')).run()
    assert not app.exception
    assert any('No month of the chosen period' in c.value for c in app.caption)
    app.slider(key='inflation_bounds').set_value((2.0, 4.0)).run()
    assert not app.exception
    app.slider(key='inflation_bounds').set_value((3.0, 3.0)).run()
    assert not app.exception
    assert any('lower threshold below the upper' in i.value for i in app.info)


def test_client_view_in_real_terms(client):
    client.segmented_control(key='terms').set_value('Real').run()
    assert not client.exception
    assert any('after inflation' in m.value for m in client.markdown)


# Euros (Section 17 of the notebook)

def test_euro_view_reproduces_the_euro_tables(app):
    app.segmented_control(key='currency').set_value('EUR').run()
    assert not app.exception
    for terms, name in (('Nominal', 'table1_euros_formatted.csv'), ('Real', 'table1_euros_real_formatted.csv')):
        app.segmented_control(key='terms').set_value(terms).run()
        assert not app.exception
        shown = performance_table(app).replace('–', '-')
        expected = pd.read_csv(OUTPUTS / name, index_col=0).fillna('-')
        assert list(shown.index) == list(expected.index)          # including the growth of one euro
        assert (shown.values == expected.values).all()
    assert any(m.label.startswith('Real value of €') for m in app.metric)


def test_method_checks_the_euro_tables(app):
    messages = [s.value for s in app.success]
    assert len(messages) == 4
    assert any('Table 1 in euros of Section 17' in m for m in messages)
    assert any('Table 1 in real euros of Section 17' in m for m in messages)


def test_currency_tab_summarizes_the_euro_results(app):
    table = pd.read_csv(OUTPUTS / 'table1_euros.csv', index_col=0)
    briefs = [m.value for m in app.markdown if 'Measured in euros, without hedging' in m.value]
    assert briefs
    assert any(f"{table.loc['Compound annual return', 'Permanent Portfolio']:.2%}" in b for b in briefs)
    assert any(f"{table.loc['Maximum drawdown', 'Permanent Portfolio']:.2%}" in b for b in briefs)
    crises = pd.read_csv(OUTPUTS / 'crisis_returns_euros.csv', index_col=0)
    rose = int((crises['Dollar against the euro'] > 0).sum())
    how_many = f'all {len(crises)}' if rose == len(crises) else f'{rose} of the {len(crises)}'
    assert any(f'rose against the euro in {how_many} equity crises' in b for b in briefs)


def test_crises_in_euros_keep_the_dates_of_section_14(app):
    app.segmented_control(key='currency').set_value('EUR').run()
    assert not app.exception
    expected = pd.read_csv(OUTPUTS / 'crisis_returns_euros.csv', index_col=0)
    table = next(t.value for t in app.table if 'Dollar against the euro' in t.value.columns
                 and 'Months, peak to trough' in t.value.columns)
    assert list(table['Equity peak']) == list(expected['Equity peak'])
    assert list(table['Permanent Portfolio']) == [f'{v:.1%}' for v in expected['Permanent Portfolio']]


@pytest.mark.parametrize('terms', ['Nominal', 'Real'])
@pytest.mark.parametrize('rule', ['Tolerance band', 'Calendar reset', 'Buy and hold'])
def test_every_rule_runs_in_euros(app, rule, terms):
    app.segmented_control(key='currency').set_value('EUR')
    app.segmented_control(key='terms').set_value(terms)
    app.segmented_control(key='rule').set_value(rule)
    app.run()
    assert not app.exception


def test_inflation_tab_in_euros_uses_euro_area_prices(app):
    app.segmented_control(key='currency').set_value('EUR').run()
    assert not app.exception
    assert any('Euro-area consumer prices rose by' in m.value for m in app.markdown)
    app.select_slider(key='period').set_value(('2016-01', '2026-10')).run()
    assert not app.exception


def test_client_view_in_euros(client):
    client.segmented_control(key='currency').set_value('EUR').run()
    assert not client.exception
    assert any('€' in m.value and 'In brief' not in m.value for m in client.markdown)
    assert any(m.label.startswith('Value of €') for m in client.metric)
