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
