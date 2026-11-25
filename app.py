"""Permanent Portfolio explorer: the Streamlit application of the Group G notebook.

Run locally with `streamlit run app.py`. Every number is computed with core.py (exact copies of the
notebook's functions) and analysis.py (the notebook's result cells as functions of the assumptions);
tests/ verifies that the default settings reproduce the notebook's tables, in nominal and in real terms, in
dollars and in euros.
"""
import os
from pathlib import Path

import altair as alt
import certifi
import numpy as np
import pandas as pd
import streamlit as st

import analysis as an
from core import (download_cpi, download_euribor, download_eurusd, download_hicp, format_table, in_euros, load_data,
                  real_returns)

# Verify HTTPS downloads against Mozilla's certificate bundle: some systems' own stores (Windows in particular)
# lack the root certificate of the ECB Data Portal
os.environ.setdefault('SSL_CERT_FILE', certifi.where())

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / 'notebook' / 'outputs'

# Links shown in the app
REPO_URL = 'https://github.com/UgoBoudjelal/Browne-s-Permanent-Portfolio-'
NOTEBOOK_PATH = 'notebook/Group_G_Permanent_Portfolio.ipynb'

PP = 'Permanent Portfolio'
VTI = 'All equity (VTI)'
RULES = ['Tolerance band', 'Calendar reset', 'Buy and hold']
DEFAULTS = {'rule': 'Tolerance band', 'band_low': 15, 'band_high': 35, 'review': 'Monthly', 'cost': 0.05,
            'cost_stress': 0.50, 'bench_equity': 60, 'amount': 10_000}

# Browne's four economic seasons and the sleeve meant for each (Browne, 1999)
SLEEVES = ['Equities (VTI)', 'Long-term Treasuries (TLT)', 'Gold (GLD)', 'Treasury bills']
SLEEVE_COLS = ['VTI', 'TLT', 'GLD', 'rf']
SEASONS = {'Equities (VTI)': 'Prosperity', 'Long-term Treasuries (TLT)': 'Deflation',
           'Gold (GLD)': 'Inflation', 'Treasury bills': 'Recession'}
SHORT = {'Equities (VTI)': 'equities', 'Long-term Treasuries (TLT)': 'long-term Treasuries', 'Gold (GLD)': 'gold',
         'Treasury bills': 'Treasury bills'}

# "Four regimes" palette, validated for colour-blind safety in both themes. Portfolios: the Permanent
# Portfolio in magenta, the stock-bond benchmark in blue, VTI in the equity sleeve's green. Sleeves: one colour
# per economic season, used on every chart of the sleeves.
PALETTE = {
    'light': {'series': ['#b03f86', '#2f7fd0', '#3a9a4b'], 'sleeves': ['#3a9a4b', '#3b6fd4', '#b07d0a', '#8a63d2'],
              'muted': '#7a837d', 'ink': '#17201b', 'soft': '#b9c0bb', 'band': '#b03f86',
              'neg': '#e66767', 'mid': '#eef0ed', 'pos': '#5598e7'},
    'dark': {'series': ['#c75298', '#4a8ee0', '#3f9f50'], 'sleeves': ['#3f9f50', '#4a8ee0', '#b98008', '#8e6fe0'],
             'muted': '#8c958f', 'ink': '#eaeeea', 'soft': '#4d5550', 'band': '#c75298',
             'neg': '#e34948', 'mid': '#2b3133', 'pos': '#2a78d6'},
}

st.set_page_config(page_title='Permanent Portfolio explorer', page_icon=':material/balance:', layout='wide')
theme_type = getattr(getattr(st.context, 'theme', None), 'type', None)
P = PALETTE['dark' if theme_type == 'dark' else 'light']


# ----------------------------------------------------------------------------------------------------------
# Data: one download for the whole sample, cached; the notebook's saved returns if the sources are busy
# ----------------------------------------------------------------------------------------------------------

EURO_COLUMNS = ['Dollars per euro', 'Euribor return', 'HICP']      # Section 17's data, as saved by the notebook


@st.cache_data(show_spinner='Downloading monthly prices (Yahoo Finance), interest rates, consumer prices and exchange '
                             'rates (FRED and the European Central Bank)...')
def live_data():
    _, rets = load_data(an.START, an.END, an.TICKERS)
    cpi = download_cpi(an.CPI_START, an.END)
    euro = pd.DataFrame({'Dollars per euro': download_eurusd(an.START, an.END),
                         'Euribor return': download_euribor(an.START, an.END),
                         'HICP': download_hicp(an.HICP_START, an.END)})
    return rets, cpi, euro


@st.cache_data(show_spinner=False)
def saved_data():
    rets = pd.read_csv(OUTPUTS / 'returns_monthly.csv', index_col=0, parse_dates=True)
    cpi = pd.read_csv(OUTPUTS / 'consumer_prices.csv', index_col=0, parse_dates=True)['CPI']
    euro = pd.read_csv(OUTPUTS / 'euro_data.csv', index_col=0, parse_dates=True)[EURO_COLUMNS]
    return rets, cpi, euro


@st.cache_data(show_spinner=False)
def notebook_table(name):
    return pd.read_csv(OUTPUTS / name, index_col=0)


def get_data():
    if os.environ.get('PP_APP_OFFLINE') == '1' or st.session_state.get('use_saved_data'):
        return *saved_data(), 'saved'
    try:
        return *live_data(), 'live'
    except Exception as exc:                       # rate limit, network error or failed calendar check
        st.session_state['use_saved_data'] = True
        st.session_state['data_error'] = f'{type(exc).__name__}: {exc}'
        return *saved_data(), 'saved'


@st.cache_data(show_spinner=False)
def rule_table(rets_p, rf_p, cost, w_bench):
    # Section 15 for the chosen period, cost, benchmark and currency (rf_p: Euribor for the euro investor)
    results, _, _ = an.simulate(rets_p, cost=cost, w_stocks_bench=w_bench, rf=rf_p)
    return an.rule_comparison(rets_p, results, cost)


# ----------------------------------------------------------------------------------------------------------
# Formatting and chart helpers
# ----------------------------------------------------------------------------------------------------------

def pct(x, d=2):
    return '-' if pd.isna(x) else f'{x:.{d}%}'


def signed_pp(x):
    return f'{100 * x:+.2f} pp'


def threshold(x):
    # An inflation threshold as written in the regime labels, e.g. 0.03 -> '3%'
    return f'{100 * x:g}%'


def table_text(df):
    # st.table renders cells as Markdown, where a lone '-' (format_table's "not applicable") becomes a bullet
    return df.map(lambda v: '–' if v == '-' else v)


def relabel(df):
    # In euros, the last row of the performance tables is the growth of one euro (as in Section 17)
    return df.rename(index={'Growth of $1': GROWTH_ROW})


def money(x):
    # An amount in the currency of the page, e.g. $10,000 or €10,000
    return f'{"-" if x < 0 else ""}{SYM}{abs(x):,.0f}'


def time_chart(wide, fmt, y_title, key, colors, log=False, under=(), over=(), height=340, y_domain=None, dashed=(),
               legend=True):
    """Lines over time with a hover crosshair; the tooltip lists every series at the hovered month."""
    names = list(wide.columns)
    data = wide.rename_axis('Date').reset_index()
    long = data.melt('Date', var_name='Series', value_name='Value')
    safe = data.copy()
    safe.columns = ['Date'] + [f's{i}' for i in range(len(names))]

    legend_spec = None
    if legend:
        # Line symbols when some series are dashed, so that the legend shows which ones
        style = dict(symbolType='stroke', symbolSize=400, symbolStrokeWidth=2) if dashed else {}
        legend_spec = alt.Legend(title=None, orient='top', direction='horizontal', labelLimit=320, **style)
    color = alt.Color('Series:N', scale=alt.Scale(domain=names, range=colors[:len(names)]), legend=legend_spec)
    if log:
        scale = alt.Scale(type='log')
    elif y_domain is not None:
        scale = alt.Scale(domain=y_domain, zero=False)
    else:
        scale = alt.Scale(zero=False)
    x = alt.X('Date:T', title=None, axis=alt.Axis(format='%Y', grid=False))
    hover = alt.selection_point(name=f'{key}_hover', fields=['Date'], on='pointerover', clear='pointerout',
                                empty=False)

    encoding = dict(x=x, y=alt.Y('Value:Q', title=y_title, scale=scale, axis=alt.Axis(format=fmt)), color=color)
    if dashed:
        # Same field and legend as the colour, so that Vega-Lite merges the two into one legend
        encoding['strokeDash'] = alt.StrokeDash('Series:N', legend=legend_spec, scale=alt.Scale(
            domain=names, range=[[6, 4] if n in dashed else [1, 0] for n in names]))
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(**encoding)
    tooltip = [alt.Tooltip('Date:T', title='Month', format='%b %Y')] + \
              [alt.Tooltip(f's{i}:Q', title=n, format=fmt) for i, n in enumerate(names)]
    targets = alt.Chart(safe).mark_rule(strokeWidth=8, opacity=0).encode(x='Date:T', tooltip=tooltip) \
        .add_params(hover)
    cursor = alt.Chart(safe).mark_rule(color=P['muted'], strokeWidth=1).encode(x='Date:T') \
        .transform_filter(hover)
    dots = lines.mark_point(filled=True, size=60, opacity=1).transform_filter(hover)
    return alt.layer(*under, lines, *over, cursor, dots, targets).properties(height=height)


def vertical_marks(dates):
    # Dashed vertical lines at the given month-ends (rebalancing dates)
    df = pd.DataFrame({'Date': pd.to_datetime(list(dates))})
    return alt.Chart(df).mark_rule(color=P['muted'], strokeDash=[3, 3], strokeWidth=1, opacity=0.8) \
        .encode(x='Date:T')


def crisis_shading(crisis_table):
    # Shaded equity peak-to-trough windows with their names, staggered on three lines
    if crisis_table.empty:
        return [], []
    df = pd.DataFrame({'start': [pd.Timestamp(v) for v in crisis_table['Equity peak']],
                       'end': [pd.Timestamp(v) for v in crisis_table['Equity trough']],
                       'label': [n.split(' (')[0] for n in crisis_table.index]})
    df['row'] = [i % 3 for i in range(len(df))]
    rects = alt.Chart(df).mark_rect(color=P['ink'], opacity=0.07).encode(x='start:T', x2='end:T')
    labels = [alt.Chart(df[df['row'] == r]).mark_text(align='left', dx=3, fontSize=10, color=P['muted'])
              .encode(x='start:T', y=alt.value(8 + 12 * r), text='label:N') for r in range(3)]
    return [rects], labels


def show_data(df, label='Show data', file_name=None):
    # Table view behind every chart (also the accessible alternative to colour)
    with st.expander(label, icon=':material/table:'):
        st.dataframe(df, width='stretch')
        if file_name:
            st.download_button('Download CSV', df.to_csv().encode('utf-8'), file_name=file_name,
                               mime='text/csv', icon=':material/download:')


def reset_assumptions():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state['period'] = (MONTHS[0], MONTHS[-1])


# ----------------------------------------------------------------------------------------------------------
# Header and data
# ----------------------------------------------------------------------------------------------------------

st.title('Permanent Portfolio explorer')
st.caption("Four assets for four economic seasons: Harry Browne's Permanent Portfolio holds 25% each in US "
           "equities, long-term Treasuries, gold and Treasury bills. Backtested on monthly data and compared with a "
           "stock–bond benchmark. Group G · 2270 Financial Modeling · Nova SBE.")

rets, cpi, euro, source = get_data()
MONTHS = list(rets.index.strftime('%Y-%m'))

# Section 16: inflation and the return panel in real terms (every series deflated by the month's inflation)
inflation, inflation_12m = an.inflation_rates(cpi, rets.index)
real_rets = real_returns(rets, inflation)

# Section 17: the panel in euros (converted at month-end exchange rates, unhedged), Euribor as the euro investor's
# risk-free rate, and euro-area inflation (HICP)
eurusd = euro['Dollars per euro'].dropna()
dollar_return = an.dollar_returns(eurusd, rets.index)
rf_eur = euro.loc[rets.index, 'Euribor return']
hicp = euro['HICP']
inflation_eur, inflation_eur_12m = an.inflation_rates(hicp, rets.index)
euro_rets = in_euros(rets, dollar_return)
euro_real_rets = real_returns(euro_rets, inflation_eur)
rf_eur_real = real_returns(rf_eur, inflation_eur)

# The four panels, keyed by (currency, terms): the returns and the risk-free rate of the performance measures
# (None: the T-bill return of the panel)
PANELS = {('USD', 'Nominal'): (rets, None), ('USD', 'Real'): (real_rets, None),
          ('EUR', 'Nominal'): (euro_rets, rf_eur), ('EUR', 'Real'): (euro_real_rets, rf_eur_real)}


def month_label(m):
    return pd.Timestamp(m + '-01').strftime('%b %Y')


# ----------------------------------------------------------------------------------------------------------
# Sidebar: the assumptions (bound to the URL, so a scenario can be shared as a link)
# ----------------------------------------------------------------------------------------------------------

with st.sidebar:
    st.header('Assumptions')
    st.caption("Every result on the page is recomputed with the notebook's code when an assumption changes. "
               "The address bar keeps the settings, so a scenario can be shared as a link.")

    rule = st.segmented_control(
        'Rebalancing rule', RULES, default=DEFAULTS['rule'], required=True, key='rule', bind='query-params',
        width='stretch',
        help='Tolerance band: restore 25% weights when any weight leaves the band at a review date (the '
             "strategy's rule). Calendar reset: restore 25% weights at every review date. Buy and hold: never "
             'rebalance.')
    review = st.segmented_control(
        'Review frequency', list(an.REVIEW_SCHEDULES), default=DEFAULTS['review'], required=True, key='review',
        bind='query-params', width='stretch', disabled=rule == 'Buy and hold',
        help='Month-ends at which the weights are checked (tolerance band) or reset (calendar).')
    band_low = st.slider('Lower bound of the band (%)', 0, 25, DEFAULTS['band_low'], 1, key='band_low',
                         bind='query-params', disabled=rule != 'Tolerance band')
    band_high = st.slider('Upper bound of the band (%)', 25, 50, DEFAULTS['band_high'], 1, key='band_high',
                          bind='query-params', disabled=rule != 'Tolerance band')
    cost_pct = st.slider('Transaction cost per side (%)', 0.0, 1.0, DEFAULTS['cost'], 0.01, format='%.2f%%',
                         key='cost', bind='query-params',
                         help='Charged on every dollar bought and sold, for the strategy and the benchmark. '
                              'The notebook uses 0.05% (base case).')
    bench_equity = st.slider('Equity share of the benchmark (%)', 0, 100, DEFAULTS['bench_equity'], 5,
                             key='bench_equity', bind='query-params',
                             help='Constant mix of VTI and US aggregate bonds (AGG), rebalanced monthly. '
                                  'The notebook uses 60/40.')
    period = st.select_slider('Investment period', options=MONTHS, value=(MONTHS[0], MONTHS[-1]),
                              format_func=month_label, key='period', bind='query-params',
                              help='The strategy starts from equal weights in the first month chosen.')
    amount = st.number_input('Amount invested', min_value=100, max_value=100_000_000,
                             value=DEFAULTS['amount'], step=1_000, key='amount', bind='query-params',
                             help='In US dollars, or in euros when EUR is selected at the top of the page.')
    with st.expander('More assumptions'):
        stress_pct = st.slider('Stress-case cost per side (%)', 0.0, 2.0, DEFAULTS['cost_stress'], 0.05,
                               format='%.2f%%', key='cost_stress', bind='query-params',
                               help='Second cost scenario shown in the performance table. The notebook uses '
                                    '0.50%.')

# Parameters in the notebook's units
cost = round(cost_pct / 100, 6)
cost_stress = round(stress_pct / 100, 6)
w_bench = bench_equity / 100
bench = an.bench_name(w_bench)
months = an.REVIEW_SCHEDULES[review]
if rule == 'Tolerance band':
    low, high, check = band_low / 100, band_high / 100, months
    rule_text = f'band {band_low}–{band_high}%, {review.lower()} review'
elif rule == 'Calendar reset':
    low, high, check = 0.25, 0.25, months
    rule_text = f'calendar reset to 25%, {review.lower()}'
else:
    low, high, check = 0.25, 0.25, []
    rule_text = 'buy and hold (never rebalanced)'

first, last = period
is_default = (rule == DEFAULTS['rule'] and review == DEFAULTS['review'] and band_low == DEFAULTS['band_low']
              and band_high == DEFAULTS['band_high'] and cost == an.COST and cost_stress == an.COST_STRESS
              and w_bench == an.W_STOCKS_BENCH and (first, last) == (MONTHS[0], MONTHS[-1]))

with st.sidebar:
    st.button("Reset to the notebook's assumptions", on_click=reset_assumptions, icon=':material/restart_alt:',
              width='stretch', disabled=is_default and amount == DEFAULTS['amount'], key='reset')

# ----------------------------------------------------------------------------------------------------------
# Status line and view
# ----------------------------------------------------------------------------------------------------------

with st.container(horizontal=True, vertical_alignment='center'):
    if source == 'live':
        st.badge('Live data', icon=':material/cloud_done:', color='green')
    else:
        st.badge("Data saved by the notebook's run", icon=':material/save:', color='orange')
    if is_default:
        st.badge("Notebook's assumptions", icon=':material/verified:', color='primary')
    else:
        st.badge('Custom assumptions', icon=':material/tune:', color='violet')
    st.space('stretch')
    currency = st.segmented_control('Currency', ['USD', 'EUR'], default='USD', required=True, key='currency',
                                    bind='query-params', label_visibility='collapsed',
                                    help='USD: results for an investor in US dollars. EUR: the same portfolios '
                                         'measured in euros by an investor in the euro area who does not hedge the '
                                         'currency, with Euribor as the risk-free rate (Section 17 of the notebook).')
    terms = st.segmented_control('Returns', ['Nominal', 'Real'], default='Nominal', required=True, key='terms',
                                 bind='query-params', label_visibility='collapsed',
                                 help='Nominal: returns as measured in the currency. Real: returns after the '
                                      'consumer-price inflation of the currency area (US CPI for dollars, euro-area '
                                      'HICP for euros), with values at the prices of the month in which the money is '
                                      'invested.')
    view = st.segmented_control('View', ['Client', 'Analyst'], default='Client', required=True, key='view',
                                bind='query-params', label_visibility='collapsed',
                                help='Client: the key results in plain language. Analyst: every table and '
                                     'analysis of the notebook.')
REAL = terms == 'Real'
EUR = currency == 'EUR'
SYM, CODE = ('€', 'EUR') if EUR else ('$', 'USD')
GROWTH_ROW = f'Growth of {SYM}1'
MONEY_FMT = ',.0f' if EUR else '$,.0f'          # Vega's number formats have no euro sign; axis titles name the euro
RF_NAME = 'Euribor' if EUR else 'Treasury bills'
# The consumer prices of the investor's currency area (Section 16 for dollars, Section 17 for euros)
if EUR:
    price_index, infl_m, infl_12m = hicp, inflation_eur, inflation_eur_12m
    PRICES, PRICE_INDEX = 'euro-area consumer prices', 'harmonised index of consumer prices of the euro area (HICP)'
    INDEX_SHORT = 'HICP'
else:
    price_index, infl_m, infl_12m = cpi, inflation, inflation_12m
    PRICES, PRICE_INDEX = 'US consumer prices', 'US consumer price index (CPI-U)'
    INDEX_SHORT = 'CPI'
PRICES_CAP = PRICES[0].upper() + PRICES[1:]
st.caption(f'Rule: {rule_text} · cost {cost:.2%} per side · benchmark {bench} VTI/AGG · '
           f'{month_label(first)} to {month_label(last)}' + (' · in euros, unhedged' if EUR else '')
           + (' · returns after inflation' if REAL else ''))

if st.session_state.get('data_error') and source == 'saved':
    with st.container(border=True):
        st.warning('Yahoo Finance, FRED or the European Central Bank did not respond, so the page uses the monthly '
                   "returns, consumer prices, exchange rates and Euribor rates saved by the notebook's own run, which "
                   'cover the same months. Live data can be retried in a minute.', title='Data source busy, try again',
                   icon=':material/cloud_off:')

        def retry():
            st.session_state['use_saved_data'] = False
            st.session_state['data_error'] = None
            live_data.clear()

        st.button('Try live data again', on_click=retry, icon=':material/refresh:', key='retry')

def panel_for(key):
    # Returns and risk-free rate of one of the four panels over the chosen period
    panel, rf = PANELS[key]
    return panel.loc[first:last], None if rf is None else rf.loc[first:last]


rets_p, rf_p = panel_for((currency, terms))      # every section below follows the choice of currency and returns
if len(rets_p) < 36:
    st.warning('Choose a period of at least 36 months: shorter samples give unreliable estimates of '
               'volatility, beta and alpha.', icon=':material/date_range:')
    st.stop()

# ----------------------------------------------------------------------------------------------------------
# Model: the notebook's calculations under the chosen assumptions
# ----------------------------------------------------------------------------------------------------------

# All four runs are kept: the chosen one drives every section, the Inflation tab compares nominal and real
# returns in the chosen currency, and the Currency tab compares dollars and euros
runs = {}
for key in PANELS:
    panel_k, rf_k = panel_for(key)
    runs[key] = an.simulate(panel_k, low, high, check, cost, cost_stress, w_bench, rf=rf_k)
results, weights_after, weights_before = runs[(currency, terms)]
results_nom, results_real = runs[(currency, 'Nominal')][0], runs[(currency, 'Real')][0]
real_p = panel_for((currency, 'Real'))[0]
nominal_usd_p, nominal_eur_p = panel_for(('USD', 'Nominal'))[0], panel_for(('EUR', 'Nominal'))[0]
perf_table, perf_vs_stocks = an.performance_tables(results, bench)
perf_real = perf_table if REAL else an.performance_tables(results_real, bench)[0]
perf_other = an.performance_tables(runs[('USD' if EUR else 'EUR', terms)][0], bench)[0]    # the other currency
levels = sorted(set(an.COST_LEVELS + [cost, cost_stress]))
with np.errstate(divide='ignore'):            # without trades the break-even cost is undefined
    cost_summary, sensitivity = an.cost_analysis(results, cost, cost_stress, levels)
record = an.rebalancing_record(results, weights_before, cost, cost_stress)
# Crises are dated as in Section 14 (US equities in dollars) and measured in the chosen currency and terms
dollar_p = dollar_return.loc[first:last]
crisis_usd, recovery_usd = an.crisis_analysis(runs[('USD', 'Nominal')][0], bench)
if (currency, terms) == ('USD', 'Nominal'):
    crisis_table, recovery_table = crisis_usd, recovery_usd
else:
    crisis_table, recovery_table = an.crises_on_dates(results, crisis_usd, bench, dollar_p if EUR else None)
rebalance_dates = list(results.index[results['turnover'] > 0])

colors3 = P['series']
names3 = [PP, bench, VTI]
growth = pd.DataFrame({PP: an.wealth_index(results['pp_net']), bench: an.wealth_index(results['bench_6040']),
                       VTI: an.wealth_index(results['stocks'])})     # value of one unit, from 1 the month before
wealth = growth * amount
start_ts = rets_p.index[0] - pd.DateOffset(months=1)        # the month-end at which the money is invested
start_label, end_label = f'{start_ts:%B %Y}', f'{rets_p.index[-1]:%B %Y}'
CUR_NOTE = ' All values are in euros, converted from US dollars at month-end exchange rates without hedging.' \
    if EUR else ''
REAL_NOTE = f' All returns are after {"euro-area" if EUR else "US"} consumer-price inflation.' if REAL else ''
NOTE = CUR_NOTE + REAL_NOTE

# Consumer prices of the currency area and the dollar against the euro over the chosen period
prices_p = price_index.loc[start_ts:rets_p.index[-1]]
price_rise = prices_p.iloc[-1] / prices_p.iloc[0] - 1
inflation_rate = (1 + price_rise) ** (an.PERIODS / len(rets_p)) - 1
inflation_p, inflation_12m_p = infl_m.loc[first:last], infl_12m.loc[first:last]
dollar_gain = (1 + dollar_p).prod() - 1

# KPI row: the Permanent Portfolio net of costs, with the difference from the frictionless benchmark
pp_m, b_m = perf_table[PP], perf_table[bench]
final_pp, final_b = amount * pp_m['Growth of $1'], amount * b_m['Growth of $1']
diff_final = final_pp - final_b
with st.container(horizontal=True):
    st.metric(f'Real value of {money(amount)}' if REAL else f'Value of {money(amount)}', money(final_pp),
              f"{'+' if diff_final >= 0 else '-'}{money(abs(diff_final))}", delta_description=f'vs {bench}',
              help=(f'At {start_label} prices' if REAL else '') + (', in euros' if EUR and REAL else
                                                                    'In euros' if EUR else '') or None,
              border=True)
    st.metric('Real compound annual return' if REAL else 'Compound annual return', pct(pp_m['Compound annual return']),
              signed_pp(pp_m['Compound annual return'] - b_m['Compound annual return']),
              delta_description=f'vs {bench}', border=True)
    st.metric('Annualized volatility', pct(pp_m['Annualized volatility']),
              signed_pp(pp_m['Annualized volatility'] - b_m['Annualized volatility']), delta_color='inverse',
              delta_description=f'vs {bench}', border=True)
    st.metric('Sharpe ratio', f"{pp_m['Sharpe ratio']:.2f}", f"{pp_m['Sharpe ratio'] - b_m['Sharpe ratio']:+.2f}",
              delta_description=f'vs {bench}', border=True)
    st.metric('Maximum drawdown', pct(pp_m['Maximum drawdown']),
              signed_pp(pp_m['Maximum drawdown'] - b_m['Maximum drawdown']), delta_color='inverse',
              delta_description=f'vs {bench}', border=True)
kpi_notes = []
if EUR:
    kpi_notes.append('in euros, every return is converted from US dollars at month-end exchange rates without hedging, '
                     'and the Sharpe ratio is measured against Euribor')
if REAL:
    kpi_notes.append(f'after inflation, every return is deflated by {PRICES} ({INDEX_SHORT}) and values are at '
                     f'{start_label} prices')
if kpi_notes:
    kpi_note = '; '.join(kpi_notes)
    st.caption(kpi_note[0].upper() + kpi_note[1:] + '.')


# ----------------------------------------------------------------------------------------------------------
# Sections (each tab is one function, shared by the client and analyst views)
# ----------------------------------------------------------------------------------------------------------

def section_overview():
    # The key results in plain sentences, written from the computed numbers
    v_m = perf_table[VTI]
    lines = [
        f"From the end of {start_label} to the end of {end_label}, {money(amount)} in the Permanent Portfolio grew "
        f"to **{money(final_pp)}**{f' at {start_label} prices' if REAL else ''}, a {'real ' if REAL else ''}"
        f"compound return of {pct(pp_m['Compound annual return'])} a year, against "
        f"{money(final_b)} ({pct(b_m['Compound annual return'])}) for the {bench} and "
        f"{money(amount * v_m['Growth of $1'])} ({pct(v_m['Compound annual return'])}) for an all-equity portfolio.",
        f"Its largest fall {'in purchasing power ' if REAL else ''}from a previous peak was "
        f"**{pct(pp_m['Maximum drawdown'], 1)}**, against "
        f"{pct(b_m['Maximum drawdown'], 1)} for the {bench} and {pct(v_m['Maximum drawdown'], 1)} for equities, "
        f"with an annualized volatility of {pct(pp_m['Annualized volatility'], 1)} against "
        f"{pct(b_m['Annualized volatility'], 1)}.",
    ]
    if not crisis_table.empty:
        better = int((crisis_table[PP] > crisis_table[bench]).sum())
        n_crises = len(crisis_table)
        worst = crisis_table[PP].idxmin()
        episode = {'Financial crisis (2007-09)': 'the financial crisis of 2007–09',
                   'Debt crises (2011)': 'the debt crises of 2011', 'Q4 2018 sell-off': 'the sell-off of late 2018',
                   'COVID-19 crash (2020)': 'the COVID-19 crash of 2020',
                   'Rate shock (2022)': 'the rate shock of 2022'}.get(worst, worst)
        how_many = f'all {n_crises}' if better == n_crises else f'{better} of the {n_crises}'
        worst_return = crisis_table.loc[worst, PP]
        tail = (f"its largest peak-to-trough loss in a crisis was {pct(-worst_return, 1)}, during {episode}"
                if worst_return < 0 else 'it held or gained value from the equity peak to the trough in every one')
        lines.append(f"It did better than the {bench} in **{how_many}** equity crises of the period; {tail}.")
    lines.append(f"Per unit of risk it earned a Sharpe ratio of **{pp_m['Sharpe ratio']:.2f}**, against "
                 f"{b_m['Sharpe ratio']:.2f} for the {bench}.")
    if REAL:
        lines.append(f"{PRICES_CAP} rose by {pct(price_rise, 1)} over the period, "
                     f"{pct(inflation_rate)} a year; every value on this page is after inflation, at {start_label} "
                     "prices.")
    else:
        lines.append(f"{PRICES_CAP} rose by {pct(price_rise, 1)} over the period, "
                     f"{pct(inflation_rate)} a year, so after inflation the Permanent Portfolio grew by "
                     f"{pct(perf_real.loc['Compound annual return', PP])} a year, against "
                     f"{pct(perf_real.loc['Compound annual return', bench])} for the {bench}. Select *Real* at the "
                     "top of the page to see every result after inflation.")
    moved = f"{'gained' if dollar_gain >= 0 else 'lost'} {pct(abs(dollar_gain), 1)} against the euro"
    real_word = 'real ' if REAL else ''
    if EUR:
        lines.append(f"Over the period the dollar {moved}; in dollars, the Permanent Portfolio returned "
                     f"{pct(perf_other.loc['Compound annual return', PP])} a year{' in real terms' if REAL else ''}, "
                     "against "
                     f"{pct(perf_other.loc['Compound annual return', bench])} for the {bench}. Every value on this "
                     "page is in euros, without hedging the currency.")
    else:
        lines.append(f"For an investor in the euro area who held the same funds without hedging the currency, the "
                     f"Permanent Portfolio returned {pct(perf_other.loc['Compound annual return', PP])} a year in "
                     f"{real_word}euros, against {pct(perf_other.loc['Compound annual return', bench])} for the "
                     f"{bench}, as the dollar {moved}. Select *EUR* at the top of the page to see every result in "
                     "euros.")
    with st.container(border=True):
        st.markdown('#### In brief')
        st.markdown('\n'.join(f'- {s}' for s in lines))
        st.caption(f'Permanent Portfolio net of {cost:.2%} costs per side; {bench} before costs. The sentences '
                   'update with the assumptions in the sidebar.' + NOTE)

    st.subheader(f'{"Real value" if REAL else "Value"} of {money(amount)} invested at the end of {start_label}')
    with st.container(horizontal=True, vertical_alignment='center'):
        st.space('stretch')
        log_scale = st.toggle('Log scale', key='log_scale')
        show_marks = st.toggle('Rebalancing dates', key='show_marks', value=True)
    shade, labels = crisis_shading(crisis_table)
    under = shade + ([vertical_marks(rebalance_dates)] if show_marks and 0 < len(rebalance_dates) <= 60 else [])
    y_title = f'Value at {start_ts:%b %Y} prices ({CODE})' if REAL else f'Portfolio value ({CODE})'
    st.altair_chart(time_chart(wealth, MONEY_FMT, y_title, 'wealth', colors3, log=log_scale,
                               under=under, over=labels, height=400),
                    alt=f'Growth of {money(amount)} in the Permanent Portfolio, the {bench} benchmark and VTI, with '
                        'the equity crises shaded')
    st.caption('Shaded: equity crises, from the equity peak to the trough. Dashed lines: months in which the '
               'portfolio was rebalanced back to 25% each.' + NOTE)
    show_data(wealth.round(2), file_name='portfolio_values.csv')

    left, right = st.columns([3, 2], gap='large')
    with left:
        st.markdown('**At a glance**')
        glance = ['Compound annual return', 'Annualized volatility', 'Sharpe ratio', 'Maximum drawdown',
                  'Growth of $1']
        st.table(table_text(format_table(relabel(perf_table.loc[glance, names3]))),
                 alt='Key measures of the three portfolios')
    with right:
        st.markdown('**What the measures mean**')
        st.markdown(
            '- *Compound return*: the constant annual rate that turns the initial amount into the final value.\n'
            '- *Volatility*: how much monthly returns swing, annualized.\n'
            f'- *Sharpe ratio*: return above {RF_NAME} per unit of volatility.\n'
            '- *Maximum drawdown*: the largest fall from a previous peak in value.\n'
            f'- *{GROWTH_ROW}*: what each {"euro" if EUR else "dollar"} invested at the start became.')


def section_performance():
    heading = {('USD', 'Nominal'): 'Performance measures', ('USD', 'Real'): 'Performance measures after inflation',
               ('EUR', 'Nominal'): 'Performance measures in euros',
               ('EUR', 'Real'): 'Performance measures in euros, after inflation'}
    basis = {('USD', 'Nominal'): '',
             ('USD', 'Real'): ' All returns are after US consumer-price inflation, and the Sharpe ratio and alpha are '
                              'measured against the real return of Treasury bills (Section 16 of the notebook).',
             ('EUR', 'Nominal'): ' All returns are in euros, converted from US dollars at month-end exchange rates '
                                 'without hedging, and the Sharpe ratio and alpha are measured against Euribor '
                                 '(Section 17 of the notebook).',
             ('EUR', 'Real'): ' All returns are in euros and after euro-area consumer-price inflation (HICP), and the '
                              'Sharpe ratio and alpha are measured against the real return of Euribor (Section 17 of '
                              'the notebook).'}
    st.subheader(heading[(currency, terms)])
    st.caption(f'{month_label(first)} to {month_label(last)}. Permanent Portfolio net of {cost:.2%} (base) and '
               f'{cost_stress:.2%} (stress) costs per side; the benchmark without and with base costs. Beta, alpha, '
               f'tracking error and information ratio are measured against the frictionless {bench}; beta is not '
               'annualized.' + basis[(currency, terms)])
    st.table(table_text(format_table(relabel(perf_table))), alt='Performance measures of the portfolios')
    with st.container(horizontal=True):
        with st.container(border=True):
            st.markdown('**Relative to the all-equity portfolio (VTI)**')
            st.table(table_text(format_table(perf_vs_stocks)), alt='Permanent Portfolio measured against VTI')
        with st.container(border=True):
            st.markdown('**Definitions**')
            st.markdown(
                '- *Average return* is the mean monthly return × 12; *compound return* is the constant annual '
                'rate that turns the initial amount into the final value.\n'
                '- *Volatility* is the standard deviation of monthly returns × √12.\n'
                f'- *Sharpe ratio*: mean monthly return in excess of {RF_NAME if EUR else "T-bills"} divided by its '
                'standard deviation, × √12.\n'
                '- *Beta*: sensitivity of the excess return to that of the benchmark. *Alpha*: the annualized '
                'excess return that beta does not explain; a *t-statistic* above about 2 in absolute value '
                'indicates significance at 5%.\n'
                '- *Tracking error*: volatility of the return difference from the benchmark. *Information '
                'ratio*: mean return difference divided by tracking error.\n'
                '- *Maximum drawdown*: the largest fall from a previous peak in value.')
    show_data(relabel(perf_table), 'Unformatted values', file_name='performance_measures.csv')


def section_risk():
    drawdowns = -(wealth.cummax() - wealth) / wealth.cummax()
    st.subheader('Drawdown from the previous peak')
    st.altair_chart(time_chart(drawdowns, '.0%', 'Drawdown', 'dd', colors3, height=300,
                               y_domain=[min(-0.05, float(drawdowns.min().min())), 0]),
                    alt='Drawdowns of the three portfolios from their running maximum')
    show_data(drawdowns.round(4), file_name='drawdowns.csv')

    left, right = st.columns(2, gap='large')
    with left:
        st.subheader('Rolling three-year volatility')
        series = {PP: results['pp_net'], bench: results['bench_6040'], VTI: results['stocks']}
        rolling = pd.DataFrame({n: series[n].rolling(36).std() * np.sqrt(an.PERIODS) for n in series}).dropna()
        st.altair_chart(time_chart(rolling, '.0%', 'Annualized volatility', 'roll', colors3, height=300),
                        alt='Rolling 36-month volatility of the three portfolios')
        st.caption('Standard deviation of the previous 36 monthly returns, annualized.' + NOTE)
    with right:
        st.subheader('Risk and return')
        assets = {PP: results['pp_net'], bench: results['bench_6040'], VTI: results['stocks'],
                  'Long-term Treasuries (TLT)': rets_p['TLT'], 'Gold (GLD)': rets_p['GLD'],
                  'Aggregate bonds (AGG)': rets_p['AGG'], 'Treasury bills': rets_p['rf']}
        rr = pd.DataFrame({'Portfolio or asset': list(assets),
                           'Annualized volatility': [assets[a].std() * np.sqrt(an.PERIODS) for a in assets],
                           'Annualized average return': [assets[a].mean() * an.PERIODS for a in assets]})
        rr['Colour'] = colors3 + [P['soft']] * 4
        # Labels to the left of points that have a close neighbour on their right
        rr['Side'] = ['left', 'right', 'left', 'left', 'right', 'below', 'right']
        points = alt.Chart(rr).mark_point(filled=True, size=110, opacity=1).encode(
            x=alt.X('Annualized volatility:Q', axis=alt.Axis(format='.0%'),
                    scale=alt.Scale(domain=[0, 1.3 * float(rr['Annualized volatility'].max())])),
            y=alt.Y('Annualized average return:Q', axis=alt.Axis(format='.0%')),
            color=alt.Color('Colour:N', scale=None),
            tooltip=['Portfolio or asset', alt.Tooltip('Annualized volatility:Q', format='.2%'),
                     alt.Tooltip('Annualized average return:Q', format='.2%')])
        label_right = points.mark_text(align='left', dx=9, dy=-8, fontSize=12).encode(
            text='Portfolio or asset:N', color=alt.value(P['ink'])).transform_filter(alt.datum.Side == 'right')
        label_left = points.mark_text(align='right', dx=-9, dy=-8, fontSize=12).encode(
            text='Portfolio or asset:N', color=alt.value(P['ink'])).transform_filter(alt.datum.Side == 'left')
        label_below = points.mark_text(align='left', dx=9, dy=12, fontSize=12).encode(
            text='Portfolio or asset:N', color=alt.value(P['ink'])).transform_filter(alt.datum.Side == 'below')
        st.altair_chart(alt.layer(points, label_right, label_left, label_below).properties(height=300),
                        alt='Annualized mean return against volatility for the portfolios and their constituents')
        st.caption('The three portfolios in colour; the constituents and the bond index in grey.' + NOTE)
    show_data(rr.drop(columns=['Colour', 'Side']).set_index('Portfolio or asset'), 'Show risk and return data')


def section_rebalancing():
    kpi = st.columns(5)
    kpi[0].metric('Rebalancing events', f"{int(cost_summary['Rebalancing events'])}",
                  help=f"Over {cost_summary['Years in sample']:.1f} years", border=True)
    kpi[1].metric('Annual turnover', pct(cost_summary['Average annual turnover']),
                  help='Average purchases plus sales per year, as a share of portfolio value', border=True)
    kpi[2].metric('Annual cost drag', pct(cost_summary['Average annual cost, base case'], 3),
                  help=f'Average annual cost at {cost:.2%} per side', border=True)
    kpi[3].metric(f'{bench} turnover', pct(cost_summary['60/40 average annual turnover']),
                  help='Average annual turnover of the benchmark, rebalanced monthly to constant weights',
                  border=True)
    be = cost_summary['Break-even cost rate']
    kpi[4].metric('Break-even cost', f'{be:.1%}' if np.isfinite(be) else '–',
                  help='Cost per side at which alpha against the benchmark falls to zero', border=True)

    st.subheader('Pre-trade weights and the tolerance band')
    w_show = weights_before.copy()
    w_show.columns = SLEEVES
    band_layer = []
    if rule == 'Tolerance band':
        band_layer.append(alt.Chart(pd.DataFrame({'low': [low], 'high': [high]})).mark_rect(
            color=P['band'], opacity=0.08).encode(y='low:Q', y2='high:Q'))
    if 0 < len(rebalance_dates) <= 60:
        band_layer.append(vertical_marks(rebalance_dates))
    st.altair_chart(time_chart(w_show, '.0%', 'Weight before rebalancing', 'weights', P['sleeves'],
                               under=band_layer, height=340),
                    alt='Weights of the four sleeves after drift, with the tolerance band and rebalancing dates')
    st.caption('Weights at each month-end after drift and before any trade, in the colour of each sleeve\'s '
               'economic season. The shaded area is the tolerance band; dashed lines mark rebalancing to 25% each.')

    st.subheader('Rebalancing record')
    if record.empty:
        st.info('The portfolio is never rebalanced under this rule.', icon=':material/info:')
    else:
        shown = record.copy()
        shown.index.name = 'Month'
        pct_cols = {c: st.column_config.NumberColumn(c, format='percent') for c in shown.columns}
        st.dataframe(shown, column_config=pct_cols, height=min(38 + 35 * len(shown), 420), width='stretch',
                     alt='Pre-trade weights, turnover and cost of each rebalancing')
        st.caption('Weights of each sleeve just before the trade, turnover (purchases plus sales as a share of '
                   'portfolio value) and the cost of the trade under the base and stress cost rates.')

    st.subheader('Sensitivity to the cost rate')
    rows_shown = ['Annual cost drag', 'Annualized average return', 'Compound annual return', 'Sharpe ratio',
                  'Annualized alpha', 'Information ratio', 'Maximum drawdown']
    st.table(table_text(format_table(sensitivity.loc[rows_shown])), alt='Performance by cost rate per side')
    st.caption(f'Cost per dollar traded, on each side of a trade. Alpha and information ratio relative to the '
               f'{bench}. Because costs enter returns linearly, alpha falls in a straight line as the cost rate '
               'rises; the break-even cost is the rate at which it reaches zero.' + NOTE)


def section_crises():
    st.subheader('How each crisis unfolded')
    if crisis_table.empty:
        st.info('None of the five crisis windows lies entirely inside the chosen period.', icon=':material/info:')
        return
    dates = list(growth.index)
    frames = []
    for ep, row in crisis_table.iterrows():
        peak = pd.Timestamp(row['Equity peak'])
        stop = dates[min(dates.index(pd.Timestamp(row['Equity trough'])) + 24, len(dates) - 1)]
        seg = growth.loc[peak:stop]
        for p in names3:
            frames.append(pd.DataFrame({'Episode': ep, 'Portfolio': p, 'Months since the equity peak':
                                        range(len(seg)), 'Value': 100 * seg[p].values / seg[p].iloc[0]}))
    paths = pd.concat(frames)
    order = list(crisis_table.index)
    small = alt.Chart(paths).mark_line(strokeWidth=2).encode(
        x=alt.X('Months since the equity peak:Q'),
        y=alt.Y('Value:Q', title='Value (equity peak = 100)', scale=alt.Scale(zero=False)),
        color=alt.Color('Portfolio:N', scale=alt.Scale(domain=names3, range=colors3),
                        legend=alt.Legend(title=None, orient='top')),
        tooltip=['Episode', 'Portfolio', 'Months since the equity peak', alt.Tooltip('Value:Q', format='.1f')],
    ).properties(width=250, height=180).facet(
        facet=alt.Facet('Episode:N', sort=order, title=None, header=alt.Header(labelFontSize=12)), columns=3)
    st.altair_chart(small, alt='Value of each portfolio from the equity peak, in each of the five crises')
    st.caption('Each panel starts at the equity peak (value 100) and runs to two years after the equity trough. '
               'The crises are dated as in Section 14 of the notebook: the lowest month-end of VTI in dollars within '
               'a search window and the highest month-end before it.' +
               (' The returns are measured over the same months, ' + ('in euros' if EUR else '') +
                (' and ' if EUR and REAL else '') + ('after inflation' if REAL else '') + '.'
                if EUR or REAL else ''))

    st.subheader('Return from the equity peak to the equity trough')
    bars = crisis_table[names3].reset_index().melt('Episode', var_name='Portfolio', value_name='Return')
    base = alt.Chart(bars).encode(
        x=alt.X('Episode:N', sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=220)),
        xOffset=alt.XOffset('Portfolio:N', sort=names3),
        y=alt.Y('Return:Q', axis=alt.Axis(format='.0%'), title='Return, peak to trough'))
    columns = base.mark_bar(cornerRadiusEnd=4, width={'band': 0.9}).encode(
        color=alt.Color('Portfolio:N', scale=alt.Scale(domain=names3, range=colors3),
                        legend=alt.Legend(title=None, orient='top')),
        tooltip=['Episode', 'Portfolio', alt.Tooltip('Return:Q', format='.1%')])
    up = base.mark_text(dy=-7, fontSize=11, color=P['ink']).encode(text=alt.Text('Return:Q', format='.1%')) \
        .transform_filter('datum.Return >= 0')
    down = base.mark_text(dy=9, fontSize=11, color=P['ink']).encode(text=alt.Text('Return:Q', format='.1%')) \
        .transform_filter('datum.Return < 0')
    zero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color=P['muted']).encode(y='y:Q')
    st.altair_chart(alt.layer(columns, zero, up, down).properties(height=340),
                    alt='Peak-to-trough returns of the three portfolios in five equity crises')

    # In euros the return table has a further column, so the two tables are stacked at full width
    left, right = (st.container(), st.container()) if EUR else st.columns([3, 2], gap='large')
    with left:
        st.markdown('**Peak-to-trough returns**')
        shown = crisis_table.copy()
        for p in shown.columns[3:]:                  # the portfolios and, in euros, the dollar against the euro
            shown[p] = [pct(v, 1) for v in crisis_table[p]]
        st.table(shown, alt='Peak-to-trough returns by episode')
        if EUR:
            st.caption('Dollar against the euro: the change in the value of one dollar in euros over the same '
                       'months.')
    with right:
        st.markdown('**Months from the equity peak to recovery**')
        st.table(recovery_table.map(lambda v: 'not recovered' if pd.isna(v) else f'{int(v)}'),
                 alt='Months from the equity peak until each portfolio regained its value')
        st.caption('0 means the portfolio stood at or above its peak value at the trough.')


def section_regimes():
    left, right = st.columns([2, 3], gap='large')
    with left:
        st.subheader('Four assets for four seasons')
        cells = pd.DataFrame({'Season': ['Prosperity', 'Inflation', 'Deflation', 'Recession'],
                              'Sleeve': ['Equities (VTI)', 'Gold (GLD)', 'Long-term Treasuries (TLT)',
                                         'Treasury bills'],
                              'x': [0, 1, 0, 1], 'y': [1, 1, 0, 0]})
        cells['Colour'] = [P['sleeves'][SLEEVES.index(s)] for s in cells['Sleeve']]
        cells['Label'] = [s.replace('Long-term Treasuries', 'Long-term\nTreasuries') for s in cells['Sleeve']]
        # Positions inside each tile (tiles span 0.96 units; the gap between them is 0.04)
        cells['x2'], cells['y2'] = cells['x'] + 0.96, cells['y'] + 0.96
        cells['mx'], cells['my'] = cells['x'] + 0.14, cells['y'] + 0.76      # colour marker
        cells['tx'] = cells['x'] + 0.08                                        # left edge of the text
        cells['sy'], cells['ly'] = cells['y'] + 0.50, cells['y'] + 0.28       # season, sleeve
        sx = alt.Scale(domain=[0, 1.96])

        def pos(xcol, ycol):
            return dict(x=alt.X(f'{xcol}:Q', axis=None, scale=sx), y=alt.Y(f'{ycol}:Q', axis=None, scale=sx))

        tiles = alt.Chart(cells).mark_rect(cornerRadius=12, opacity=0.22).encode(
            **pos('x', 'y'), x2='x2:Q', y2='y2:Q', color=alt.Color('Colour:N', scale=None))
        marker = alt.Chart(cells).mark_square(size=180, opacity=1).encode(
            **pos('mx', 'my'), color=alt.Color('Colour:N', scale=None))
        season = alt.Chart(cells).mark_text(align='left', fontSize=17, fontWeight='bold', color=P['ink']).encode(
            **pos('tx', 'sy'), text='Season:N')
        sleeve = alt.Chart(cells).mark_text(align='left', baseline='top', lineBreak='\n', fontSize=11,
                                            color=P['muted']).encode(**pos('tx', 'ly'), text='Label:N')
        st.altair_chart(alt.layer(tiles, marker, season, sleeve).properties(height=330),
                        alt="Browne's four economic seasons and the sleeve meant for each")
        st.caption('Browne (1999) argues that each asset does well in a different economic season, so holding all '
                   'four in equal parts means the portfolio does not depend on predicting which season comes next. '
                   'The Inflation tab tests the inflation and deflation seasons against the data.')
    with right:
        st.subheader('Which sleeve led each year')
        cal = an.calendar_returns(rets_p[SLEEVE_COLS])
        counts = rets_p.groupby(rets_p.index.year).size()
        cal.columns = SLEEVES
        best = cal.idxmax(axis=1)
        lead = pd.DataFrame({'Year': [f'{y}*' if counts[y] < 12 else str(y) for y in cal.index],
                             'Sleeve': best.values, 'Return': cal.max(axis=1).values, 'Height': 1.0})
        # Full-height bars, one per year, in the colour of the year's best sleeve
        strip = alt.Chart(lead).mark_bar(cornerRadius=4, width={'band': 0.86}).encode(
            x=alt.X('Year:O', sort=list(lead['Year']), title=None,
                    axis=alt.Axis(labelAngle=0, ticks=False, labelExpr="'’' + substring(datum.label, 2)")),
            y=alt.Y('Height:Q', axis=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('Sleeve:N', scale=alt.Scale(domain=SLEEVES, range=P['sleeves']), legend=None),
            tooltip=['Year', 'Sleeve', alt.Tooltip('Return:Q', format='.1%')]).properties(height=70)
        st.altair_chart(strip, alt='The best-performing sleeve in each calendar year, coloured as in the '
                                   'four-seasons graphic')
        wins = best.value_counts()
        st.caption('Each bar is a calendar year, coloured by the sleeve with the best return that year (colours as '
                   'in the four seasons). '
                   'Best calendar-year return among the four sleeves: '
                   + '; '.join(f'{SHORT[s]} {int(wins.get(s, 0))} {"year" if wins.get(s, 0) == 1 else "years"}'
                               for s in SLEEVES)
                   + '. * Partial year.')

        st.subheader(f"Calendar-year {'real ' if REAL else ''}returns of each sleeve{' in euros' if EUR else ''}")
        grid = st.columns(2)
        for i, s in enumerate(SLEEVES):
            df = pd.DataFrame({'Year': lead['Year'], 'Return': cal[s].values})
            chart = alt.Chart(df).mark_bar(cornerRadiusEnd=3, color=P['sleeves'][i]).encode(
                x=alt.X('Year:O', sort=list(df['Year']), title=None, axis=alt.Axis(labels=False, ticks=False)),
                y=alt.Y('Return:Q', axis=alt.Axis(format='.0%'), title=None),
                tooltip=['Year', alt.Tooltip('Return:Q', format='.1%')]).properties(
                height=130, title=alt.TitleParams(f'{SEASONS[s]} · {s}', fontSize=12, anchor='start'))
            with grid[i % 2]:
                st.altair_chart(chart, alt=f'Calendar-year returns of {s}')
        show_data(cal.set_axis(lead['Year']), 'Show calendar-year returns of the sleeves',
                  file_name='sleeve_calendar_returns.csv')


def section_rules():
    st.subheader('The strategy under alternative rebalancing rules')
    st.caption(f'Each rule is simulated with the same engine over the chosen period, net of {cost:.2%} costs per '
               f'side; alpha and the information ratio are relative to the {bench}. The strategy uses Browne\'s '
               'tolerance band of 15–35%, reviewed monthly.' + NOTE)
    rules_tbl = rule_table(rets_p, rf_p, cost, w_bench).copy()

    # Locate the current settings in the grid, or add them as a row of their own
    if rule == 'Buy and hold':
        current = 'Buy and hold'
    elif rule == 'Calendar reset':
        current = f'Calendar reset, {review.lower()}'
    else:
        current = f'Band {band_low}-{band_high}%, {review.lower()} review'
    matches = [r for r in rules_tbl.index if r.replace(' (strategy)', '') == current]
    if matches:
        current = matches[0]
    else:
        row = {'Rebalances': int((results['turnover'] > 0).sum()),
               'Annual turnover': cost_summary['Average annual turnover']}
        row.update({k: perf_table.loc[k, PP] for k in an.MEASURES_SHOWN})
        current = f'{current} (current)'
        rules_tbl.loc[current] = row
    rules_tbl['Rule type'] = ['Tolerance band' if r.startswith('Band') else
                              'Calendar reset' if r.startswith('Calendar') else 'Buy and hold'
                              for r in rules_tbl.index]

    plot_df = rules_tbl.reset_index()
    plot_df['Highlight'] = ['Strategy' if '(strategy)' in r else 'Current settings' if r == current else 'Other'
                            for r in plot_df['Rule']]
    hl_domain = ['Strategy', 'Current settings', 'Other']
    hl_range = [P['series'][0], P['ink'], P['soft']]
    bench_sharpe = perf_table.loc['Sharpe ratio', bench]
    pts = alt.Chart(plot_df).mark_point(filled=True, opacity=1).encode(
        x=alt.X('Annual turnover:Q', axis=alt.Axis(format='.0%'), title='Average annual turnover'),
        y=alt.Y('Sharpe ratio:Q', scale=alt.Scale(zero=False), title='Sharpe ratio'),
        shape=alt.Shape('Rule type:N', legend=alt.Legend(title=None, orient='top')),
        color=alt.Color('Highlight:N', scale=alt.Scale(domain=hl_domain, range=hl_range),
                        legend=alt.Legend(title=None, orient='top')),
        size=alt.when(alt.datum.Highlight != 'Other').then(alt.value(170)).otherwise(alt.value(80)),
        tooltip=['Rule', 'Rebalances', alt.Tooltip('Annual turnover:Q', format='.2%'),
                 alt.Tooltip('Sharpe ratio:Q', format='.2f'), alt.Tooltip('Maximum drawdown:Q', format='.2%'),
                 alt.Tooltip('Annualized alpha:Q', format='.2%'), alt.Tooltip('Alpha t-statistic:Q', format='.2f')])
    tags = alt.Chart(plot_df[plot_df['Highlight'] != 'Other']).mark_text(align='left', dx=10, dy=-8, fontSize=12,
                                                                         color=P['ink']).encode(
        x='Annual turnover:Q', y='Sharpe ratio:Q', text='Rule:N')
    bench_line = alt.Chart(pd.DataFrame({'y': [bench_sharpe], 'label': [f'{bench} benchmark']}))
    bench_rule = bench_line.mark_rule(color=P['muted'], strokeDash=[4, 4]).encode(
        y=alt.Y('y:Q', title='Sharpe ratio'))
    bench_text = bench_line.mark_text(align='left', dx=4, dy=-7, fontSize=11, color=P['muted']).encode(
        y='y:Q', x=alt.value(4), text='label:N')
    st.altair_chart(alt.layer(bench_rule, bench_text, pts, tags).properties(height=380),
                    alt='Sharpe ratio against annual turnover for thirteen rebalancing rules')

    shown = rules_tbl.drop(columns=['Rule type']).copy()
    fmt_cols = {}
    for col in shown.columns:
        if col in ['Annual turnover', 'Annualized average return', 'Compound annual return',
                   'Annualized volatility', 'Maximum drawdown', 'Annualized alpha']:
            fmt_cols[col] = st.column_config.NumberColumn(col, format='percent')
        elif col != 'Rebalances':
            fmt_cols[col] = st.column_config.NumberColumn(col, format='%.2f')
    st.dataframe(shown, column_config=fmt_cols, width='stretch', height=38 + 35 * len(shown),
                 alt='Performance of the strategy under each rebalancing rule')


def section_calendar():
    st.subheader(f"Calendar-year {'real ' if REAL else ''}returns{' in euros' if EUR else ''}")
    cal_src = pd.DataFrame({PP: results['pp_net'], bench: results['bench_6040'], VTI: results['stocks'],
                            'TLT': rets_p['TLT'], 'GLD': rets_p['GLD'], 'AGG': rets_p['AGG'],
                            'T-bills': rets_p['rf']})
    cal = an.calendar_returns(cal_src)
    # Columns shown for reference: the inflation of the currency area and, in euros, the dollar against the euro
    references = {f'Inflation ({INDEX_SHORT})': inflation_p}
    if EUR:
        references['Dollar in euros'] = dollar_p
    for name in references:
        cal[name] = an.calendar_returns(references[name].to_frame('x'))['x']
    counts = cal_src.groupby(cal_src.index.year).size()
    cal.index = [f'{y}*' if counts[y] < 12 else str(y) for y in cal.index]
    heat = cal.rename_axis('Year').reset_index().melt('Year', var_name='Series', value_name='Return')
    returns_only = heat[~heat['Series'].isin(list(references))]
    lim = float(min(0.4, returns_only['Return'].abs().max()))
    x = alt.X('Series:N', sort=list(cal.columns), title=None,
              axis=alt.Axis(orient='top', labelAngle=0, labelLimit=120, labelOverlap=False,
                            labelExpr=f"datum.label == '{PP}' ? ['Permanent', 'Portfolio'] : datum.label"))
    y = alt.Y('Year:N', sort=list(cal.index), title=None)
    rects = alt.Chart(returns_only).mark_rect(cornerRadius=3).encode(
        x=x, y=y,
        color=alt.Color('Return:Q', scale=alt.Scale(domain=[-lim, 0, lim], range=[P['neg'], P['mid'], P['pos']],
                                                    clamp=True, interpolate='lab'),
                        legend=alt.Legend(title='Return', format='.0%', orient='bottom', direction='horizontal',
                                          gradientLength=220)),
        tooltip=['Year', 'Series', alt.Tooltip('Return:Q', format='.1%')])
    # The reference columns are shown on a neutral background rather than on the scale of returns
    rects_reference = alt.Chart(heat[heat['Series'].isin(list(references))]).mark_rect(
        cornerRadius=3, color=P['mid'], stroke=P['soft'], strokeWidth=1).encode(
        x=x, y=y, tooltip=['Year', 'Series', alt.Tooltip('Return:Q', title='Change', format='.1%')])
    text = alt.Chart(heat).mark_text(fontSize=11, color=P['ink']).encode(
        x=x, y=y, text=alt.Text('Return:Q', format='.1%'))
    st.altair_chart(alt.layer(rects, rects_reference, text).properties(height=24 * len(cal) + 80),
                    alt='Calendar-year returns of the portfolios and of each asset, with the inflation of each year')
    st.caption('Portfolio returns net of costs for the Permanent Portfolio. * Partial year: only the months inside '
               f'the chosen period. Blue years are gains, red years losses. Inflation ({INDEX_SHORT}): the change in '
               f'{PRICES} over the same months, for reference.'
               + (' Dollar in euros: the change in the value of one dollar in euros, which compounds with the dollar '
                  'return of every asset to give its return in euros.' if EUR else '') + NOTE)
    show_data(cal, file_name='calendar_returns.csv')


def section_inflation():
    # Section 16 of the notebook for the chosen period: purchasing power, real returns and inflation regimes. In
    # euros, the same analysis with euro-area inflation (the HICP of Section 17).
    r_pp, r_b = perf_real[PP], perf_real[bench]
    kpi = st.columns(4)
    kpi[0].metric('Rise in consumer prices', pct(price_rise, 1), border=True,
                  help=f'{PRICE_INDEX[0].upper() + PRICE_INDEX[1:]} from {start_label} to {end_label}')
    kpi[1].metric('Average inflation', pct(inflation_rate), border=True,
                  help='Compound annual rate of inflation over the period')
    kpi[2].metric('Real compound return', pct(r_pp['Compound annual return']),
                  signed_pp(r_pp['Compound annual return'] - r_b['Compound annual return']),
                  delta_description=f'vs {bench}', border=True,
                  help='Compound annual return of the Permanent Portfolio after inflation')
    kpi[3].metric('Largest real drawdown', pct(r_pp['Maximum drawdown']),
                  signed_pp(r_pp['Maximum drawdown'] - r_b['Maximum drawdown']), delta_color='inverse',
                  delta_description=f'vs {bench}', border=True,
                  help='The largest loss of purchasing power of the Permanent Portfolio from a previous peak')
    brief = st.container(border=True)          # filled at the end, once the regimes are computed

    st.subheader('Keeping ahead of inflation')
    values = amount * pd.DataFrame({PP: an.wealth_index(results_nom['pp_net']),
                                    bench: an.wealth_index(results_nom['bench_6040']),
                                    VTI: an.wealth_index(results_nom['stocks'])})
    values[PRICES_CAP] = amount * prices_p / prices_p.iloc[0]
    st.altair_chart(time_chart(values, MONEY_FMT, f'Value ({CODE})', 'keep', colors3 + [P['muted']],
                               dashed=[PRICES_CAP], height=360),
                    alt=f'Nominal value of {money(amount)} in each portfolio against the amount needed to keep pace '
                        'with consumer prices')
    st.caption(f'Value in {"euros" if EUR else "US dollars"} of {money(amount)} invested at the end of {start_label}. '
               'The dashed line is the amount that keeps the purchasing power of the initial investment: a portfolio '
               'above it has grown in real terms.' + CUR_NOTE)
    show_data(values.round(2), file_name='purchasing_power.csv')

    st.subheader('Inflation over the previous twelve months')
    bounds_pct = st.slider('Thresholds of the inflation regimes (%)', 0.0, 6.0, (1.0, 3.0), 0.5, format='%.1f%%',
                           key='inflation_bounds', bind='query-params',
                           help='A month is in the low regime if inflation over the previous twelve months is at '
                                'or below the lower threshold, and in the high regime if it is above the upper one. '
                                'The notebook uses 1% and 3%.')
    lo, hi = bounds_pct[0] / 100, bounds_pct[1] / 100
    if hi <= lo:
        st.info('Choose a lower threshold below the upper threshold.', icon=':material/tune:')
        with brief:
            st.markdown('#### In brief')
            st.markdown(f'{PRICES_CAP} rose by {pct(price_rise, 1)} from {start_label} to {end_label}.')
        return
    regime_table, regime = an.inflation_regimes(real_p, results_real, inflation_12m_p, (lo, hi), bench)
    labels = list(regime_table.columns)
    months = regime_table.loc['Months']

    y_min = min(float(inflation_12m_p.min()), lo) - 0.01
    y_max = max(float(inflation_12m_p.max()), hi) + 0.01
    bands = pd.DataFrame({'y': [y_min, hi], 'y2': [lo, y_max], 'Colour': [P['sleeves'][1], P['sleeves'][2]],
                          'Label': [f'Low inflation ({threshold(lo)} or less)',
                                    f'High inflation (above {threshold(hi)})'],
                          'Middle': [(y_min + lo) / 2, (hi + y_max) / 2]})
    band_rects = alt.Chart(bands).mark_rect(opacity=0.13).encode(y='y:Q', y2='y2:Q',
                                                                   color=alt.Color('Colour:N', scale=None))
    band_text = alt.Chart(bands).mark_text(align='left', dx=6, fontSize=11, color=P['muted']).encode(
        x=alt.value(0), y='Middle:Q', text='Label:N')
    st.altair_chart(time_chart(inflation_12m_p.to_frame('Inflation over the previous 12 months'), '.1%',
                               'Inflation, previous 12 months', 'infl', [P['ink']], under=[band_rects, band_text],
                               height=260, y_domain=[y_min, y_max], legend=False),
                    alt=f'{PRICES_CAP}: inflation over the previous twelve months, with the low and high inflation '
                        'regimes shaded')
    st.caption(f'Change in {PRICES} ({INDEX_SHORT}) over the twelve months to each month-end. The shaded bands are '
               'the low and high inflation regimes, coloured as the deflation and inflation seasons of the four-'
               'seasons graphic.')

    st.subheader('Real returns by inflation regime')
    shown = [k for k in labels if months[k] > 0]
    rows = [n for n in regime_table.index if n not in ('Months', 'Average 12-month inflation')]
    heading = {k: f'{k.split(" (")[0]} · {int(months[k])} months' for k in shown}
    heat = regime_table.loc[rows, shown].rename(columns=heading).rename_axis('Series').reset_index().melt(
        'Series', var_name='Regime', value_name='Real return')
    lim = float(min(0.25, heat['Real return'].abs().max()))
    base = alt.Chart(heat).encode(
        x=alt.X('Regime:N', sort=list(heading.values()), title='Inflation over the previous twelve months',
                axis=alt.Axis(orient='top', labelAngle=0, labelLimit=240, labelOverlap=False,
                              titleFontWeight='normal', titleColor=P['muted'])),
        y=alt.Y('Series:N', sort=rows, title=None, axis=alt.Axis(labelLimit=220)))
    rects = base.mark_rect(cornerRadius=3).encode(
        color=alt.Color('Real return:Q', scale=alt.Scale(domain=[-lim, 0, lim], range=[P['neg'], P['mid'], P['pos']],
                                                         clamp=True, interpolate='lab'),
                        legend=alt.Legend(title='Real return', format='.0%', orient='right')),
        tooltip=['Series', 'Regime', alt.Tooltip('Real return:Q', format='.1%')])
    text = base.mark_text(fontSize=12, color=P['ink']).encode(text=alt.Text('Real return:Q', format='.1%'))
    st.altair_chart((rects + text).properties(height=36 * len(rows) + 50),
                    alt='Annualized real return of each sleeve and portfolio in months of low, moderate and high '
                        'inflation')
    avg = regime_table.loc['Average 12-month inflation']
    st.caption(f'Annualized compound return after inflation over the months of each regime, which need not be '
               f'consecutive. Inflation over the previous twelve months: low {threshold(lo)} or less, moderate '
               f'{threshold(lo)} to {threshold(hi)}, high above {threshold(hi)}; its average in each regime: '
               + '; '.join(f'{k.split(" (")[0].lower()} {pct(avg[k])}' for k in shown) + '.'
               + (' No month of the chosen period falls in the '
                  + ' or '.join(k.split(' (')[0].lower() for k in labels if months[k] == 0) + ' regime.'
                  if len(shown) < len(labels) else ''))
    show_data(regime_table, 'Show the regime table', file_name='inflation_regimes.csv')

    # Calendar years: inflation from December to December and the real return of each sleeve
    cal = an.calendar_real_returns(real_p, results_real, inflation_p, bench)
    counts = real_p.groupby(real_p.index.year).size()
    full = [y for y in cal.index if counts[y] == 12]

    # The plain-language summary, written from the numbers above
    notes = [f"{PRICES_CAP} rose by **{pct(price_rise, 1)}** from {start_label} to {end_label}, "
             f"{pct(inflation_rate)} a year: {money(amount)} at the end buys what {money(amount / (1 + price_rise))} "
             f"bought at the start. After inflation, the Permanent Portfolio grew by "
             f"**{pct(r_pp['Compound annual return'])}** a year, against {pct(r_b['Compound annual return'])} for "
             f"the {bench}."]
    high, low_k = labels[2], labels[0]
    if months[high] > 0:
        t = regime_table[high]
        positive = [s for s in SLEEVES if t[s] > 0]
        best = t[SLEEVES].idxmax()
        only = positive[0] if len(positive) == 1 else None
        lead = (f"{SHORT[only][0].upper() + SHORT[only][1:]} {'was' if only == 'Gold (GLD)' else 'were'} the only "
                f"sleeve with a positive real return ({pct(t[only], 1)} a year)" if only else
                f"The best sleeve was {SHORT[best]} ({pct(t[best], 1)} a year)")
        notes.append(f"In the {int(months[high])} months in which inflation over the previous year exceeded "
                     f"{threshold(hi)}, the Permanent Portfolio returned **{pct(t[PP], 1)}** a year after inflation, "
                     f"against {pct(t[bench], 1)} for the {bench}. {lead}.")
    if months[low_k] > 0:
        t = regime_table[low_k]
        best = t[SLEEVES].idxmax()
        tlt = '' if best == 'Long-term Treasuries (TLT)' else \
            f", while long-term Treasuries returned {pct(t['Long-term Treasuries (TLT)'], 1)}"
        notes.append(f"In the {int(months[low_k])} months with inflation of {threshold(lo)} or less, the best sleeve "
                     f"was {SHORT[best]} ({pct(t[best], 1)} a year after inflation){tlt}.")
    if len(full) >= 2:
        top = sorted(cal.loc[full, 'Inflation'].nlargest(2).index)
        two = (1 + cal.loc[top]).prod() - 1
        notes.append(f"In {top[0]} and {top[1]}, the two calendar years with the highest inflation "
                     f"({pct(cal.loc[top[0], 'Inflation'], 1)} and {pct(cal.loc[top[1], 'Inflation'], 1)}), the "
                     f"Permanent Portfolio lost {pct(-two[PP], 1)} of its purchasing power over the two years, against "
                     f"{pct(-two[bench], 1)} for the {bench}; gold returned {pct(two['Gold (GLD)'], 1)} and long-term "
                     f"Treasuries {pct(two['Long-term Treasuries (TLT)'], 1)} after inflation."
                     if two[PP] < 0 and two[bench] < 0 else
                     f"In {top[0]} and {top[1]}, the two calendar years with the highest inflation, the Permanent "
                     f"Portfolio returned {pct(two[PP], 1)} after inflation over the two years, against "
                     f"{pct(two[bench], 1)} for the {bench}.")
    bills = regime_table.loc['Treasury bills', shown]
    if (bills < 0).all():
        notes.append(f"{'US Treasury bills, measured in euros,' if EUR else 'Treasury bills'} lost purchasing power "
                     "in every inflation regime.")
    with brief:
        st.markdown('#### In brief')
        st.markdown('\n'.join(f'- {s}' for s in notes))
        st.caption(f'Inflation regimes: {threshold(lo)} and {threshold(hi)} over the previous twelve months. '
                   + ('This tab applies the analysis of Section 16 of the notebook to euro-area inflation and to '
                      'returns in euros (Section 17); Section 16 discusses the results for a US investor. '
                      if EUR else 'Section 16 of the notebook discusses these results; ')
                   + f'{"E" if EUR else "e"}rb and Harvey (2013) find that gold has been an unreliable inflation hedge '
                   'over the horizons that matter to most investors.')

    if len(full) >= 3:
        st.subheader('Each calendar year: inflation and the real return')
        names = SLEEVES + [PP]
        source_col = {'Equities (VTI)': VTI}
        corr = {s: cal.loc[full, 'Inflation'].corr(cal.loc[full, source_col.get(s, s)]) for s in names}
        panels = {s: f'{s.split(" (")[0]} · r = {corr[s]:.2f}' for s in names}
        dots = pd.concat([pd.DataFrame({'Year': [str(y) for y in full], 'Series': s, 'Panel': panels[s],
                                        'Inflation': cal.loc[full, 'Inflation'].values,
                                        'Real return': cal.loc[full, source_col.get(s, s)].values}) for s in names])
        point = alt.Chart().mark_point(filled=True, size=55, opacity=0.9).encode(
            x=alt.X('Inflation:Q', axis=alt.Axis(format='.0%', tickCount=4), title='Inflation that year'),
            y=alt.Y('Real return:Q', axis=alt.Axis(format='.0%'), title='Real return that year'),
            color=alt.Color('Series:N', scale=alt.Scale(domain=names, range=P['sleeves'] + [P['series'][0]]),
                            legend=None),
            tooltip=['Year', 'Series', alt.Tooltip('Inflation:Q', format='.1%'),
                     alt.Tooltip('Real return:Q', format='.1%')])
        fit = alt.Chart().transform_regression('Inflation', 'Real return').mark_line(
            color=P['muted'], strokeDash=[5, 4]).encode(x='Inflation:Q', y='Real return:Q')
        zero = alt.Chart().mark_rule(color=P['soft']).encode(y=alt.datum(0))
        grid = alt.layer(zero, fit, point, data=dots).properties(width=220, height=170).facet(
            facet=alt.Facet('Panel:N', sort=[panels[s] for s in names], title=None,
                            header=alt.Header(labelFontSize=12, labelLimit=240)), columns=3)
        st.altair_chart(grid, alt='Calendar-year real return of each sleeve and of the Permanent Portfolio against '
                                  'the inflation of the same year')
        st.caption(f'Each dot is a full calendar year ({full[0]}–{full[-1]}): inflation from December to December on '
                   'the horizontal axis and the return after inflation on the vertical axis; the dashed line is a '
                   'least-squares fit and r the correlation between the two. A downward slope means the asset lost '
                   'purchasing power in years of higher inflation.')
        show_data(cal, 'Show calendar-year inflation and real returns', file_name='calendar_real_returns.csv')


def section_currency():
    # Section 17 of the notebook for the chosen period: the same portfolios measured in euros, without hedging.
    # Nominal returns in both currencies, so that the comparison isolates the exchange rate.
    res_usd, res_eur = runs[('USD', 'Nominal')][0], runs[('EUR', 'Nominal')][0]
    p_usd = an.performance_tables(res_usd, bench)[0]
    p_eur = an.performance_tables(res_eur, bench)[0]
    kpi = st.columns(4)
    kpi[0].metric('Dollar against the euro', f'{dollar_gain:+.1%}', border=True,
                  help=f'Change in the value of one US dollar in euros from the end of {start_label} to the end of '
                       f'{end_label}')
    for col, (label, row, inverse) in zip(kpi[1:], [('Return in euros', 'Compound annual return', False),
                                                    ('Volatility in euros', 'Annualized volatility', True),
                                                    ('Drawdown in euros', 'Maximum drawdown', True)]):
        col.metric(label, pct(p_eur.loc[row, PP]), signed_pp(p_eur.loc[row, PP] - p_usd.loc[row, PP]),
                   delta_color='inverse' if inverse else 'normal', delta_description='vs in dollars', border=True,
                   help=f'{row} of the Permanent Portfolio measured in euros, compared with its value in dollars')
    brief = st.container(border=True)          # filled at the end, once the crises are computed
    if REAL:
        st.caption('This tab compares returns before inflation, to isolate the effect of the exchange rate; the '
                   'table at the bottom also shows them after the inflation of each currency area.')

    st.subheader('The same investment in euros and in dollars')
    values = amount * pd.DataFrame({f'{PP} in euros': an.wealth_index(res_eur['pp_net']),
                                    f'{bench} in euros': an.wealth_index(res_eur['bench_6040']),
                                    f'{PP} in dollars': an.wealth_index(res_usd['pp_net']),
                                    f'{bench} in dollars': an.wealth_index(res_usd['bench_6040'])})
    in_dollars = [c for c in values if c.endswith('in dollars')]
    st.altair_chart(time_chart(values, ',.0f', 'Value (euros or dollars)', 'twocur', colors3[:2] * 2,
                               dashed=in_dollars, height=380),
                    alt=f'Value of {amount:,.0f} euros and of {amount:,.0f} dollars invested in the Permanent '
                        f'Portfolio and in the {bench}')
    st.caption(f'Solid lines: €{amount:,.0f} invested at the end of {start_label}, measured in euros. Dashed lines: '
               f'${amount:,.0f} invested at the same date, measured in dollars. The euro investor holds the same '
               'dollar funds and converts at month-end exchange rates, without hedging.')
    show_data(values.round(2), file_name='values_in_two_currencies.csv')

    st.subheader('The dollar against the euro')
    rate = (1 / eurusd).loc[start_ts:rets_p.index[-1]].to_frame('Euros per US dollar')
    shade, labels = crisis_shading(crisis_usd)
    st.altair_chart(time_chart(rate, '.2f', 'Euros per US dollar', 'fx', [P['ink']], under=shade, over=labels,
                               height=260, legend=False),
                    alt='Euros per US dollar at each month-end, with the equity crises shaded')
    st.caption('Euros per US dollar at each month-end (FRED, DEXUSEU: noon buying rates in New York). When the line '
               'rises, the dollar gains against the euro and dollar assets gain value in euros. Shaded: the equity '
               'crises of Section 14, from the equity peak to the trough.')

    crises_eur = an.crises_on_dates(res_eur, crisis_usd, bench, dollar_p)[0] if not crisis_usd.empty else None
    if crises_eur is not None:
        st.subheader('Each equity crisis in euros and in dollars')
        both = pd.DataFrame({'Equity peak': crisis_usd['Equity peak'], 'Equity trough': crisis_usd['Equity trough'],
                             f'{PP}, in dollars': crisis_usd[PP], f'{PP}, in euros': crises_eur[PP],
                             f'{bench}, in dollars': crisis_usd[bench], f'{bench}, in euros': crises_eur[bench],
                             'Dollar against the euro': crises_eur['Dollar against the euro']})
        shown = both.copy()
        for c in shown.columns[2:]:
            shown[c] = [pct(v, 1) for v in both[c]]
        st.table(shown, alt='Peak-to-trough return of each portfolio in dollars and in euros, and the change in the '
                            'dollar, in each equity crisis')
        st.caption('Return from the equity peak to the equity trough, with the crises dated as in Section 14 of the '
                   'notebook. Dollar against the euro: the change in the value of one dollar in euros over the same '
                   'months.')

    st.subheader('Risk in each currency')
    effects = an.currency_effects(nominal_usd_p, res_usd, nominal_eur_p, res_eur, dollar_p, bench)
    vols = effects[['Volatility in dollars', 'Volatility in euros']].rename(
        columns=lambda c: c.replace('Volatility in ', 'In ')).rename_axis('Series').reset_index().melt(
        'Series', var_name='Measured', value_name='Volatility')
    order = list(effects.index)
    bars = alt.Chart(vols).mark_bar(cornerRadiusEnd=3, height={'band': 0.8}).encode(
        y=alt.Y('Series:N', sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        yOffset=alt.YOffset('Measured:N', sort=['In dollars', 'In euros']),
        x=alt.X('Volatility:Q', axis=alt.Axis(format='.0%'), title='Annualized volatility'),
        color=alt.Color('Measured:N', scale=alt.Scale(domain=['In dollars', 'In euros'], range=[P['soft'], P['ink']]),
                        legend=alt.Legend(title=None, orient='top')),
        tooltip=['Series', 'Measured', alt.Tooltip('Volatility:Q', format='.2%')])
    st.altair_chart(bars.properties(height=46 * len(order) + 40),
                    alt='Annualized volatility of each asset and portfolio, in dollars and in euros')
    shown = effects.copy()
    for c in shown.columns:
        shown[c] = [f'{v:.2f}' if c.startswith('Correlation') else pct(v) for v in effects[c]]
    st.table(shown, alt='Compound return and volatility in each currency, and correlation with the dollar')
    st.caption('Correlation with the dollar: between the monthly dollar return of each series and the return of the '
               'dollar against the euro. A negative correlation means that the dollar tended to rise against the euro '
               'when the series fell, which lowers its volatility in euros; a series that does not move with the '
               'dollar, such as Treasury bills, takes on the volatility of the exchange rate.')

    st.subheader('Dollars and euros, nominal and real')
    tables = {'Dollars': p_usd, 'Dollars, real': an.performance_tables(runs[('USD', 'Real')][0], bench)[0],
              'Euros': p_eur, 'Euros, real': an.performance_tables(runs[('EUR', 'Real')][0], bench)[0]}
    view_table = an.perspective_table(tables, bench)
    left, right = st.columns(2, gap='large')
    for col, p in zip((left, right), (PP, bench)):
        with col:
            st.markdown(f'**{p}**')
            part = view_table[p]
            if p == bench:
                part = part.drop(['Annualized alpha', 'Alpha t-statistic'])
            st.table(table_text(format_table(part)), alt=f'Main measures of the {p} in dollars and in euros')
    st.caption('Real: after US consumer-price inflation for dollars and after euro-area inflation (HICP) for euros. '
               'The Sharpe ratio and alpha are measured against Treasury bills in dollars and against Euribor in '
               f'euros; alpha relative to the {bench}.')

    # The plain-language summary, written from the numbers above
    moved = f"{'gained' if dollar_gain >= 0 else 'lost'} **{pct(abs(dollar_gain), 1)}** against the euro"
    falling = nominal_usd_p['VTI'] < 0
    notes = [f"From the end of {start_label} to the end of {end_label}, the dollar {moved}. Measured in euros, "
             f"without hedging, the Permanent Portfolio returned **{pct(p_eur.loc['Compound annual return', PP])}** a "
             f"year, against {pct(p_usd.loc['Compound annual return', PP])} in dollars, and the {bench} "
             f"{pct(p_eur.loc['Compound annual return', bench])}, against "
             f"{pct(p_usd.loc['Compound annual return', bench])}.",
             f"In euros, the volatility of the Permanent Portfolio was {pct(p_eur.loc['Annualized volatility', PP])} "
             f"({pct(p_usd.loc['Annualized volatility', PP])} in dollars) and its largest drawdown "
             f"**{pct(p_eur.loc['Maximum drawdown', PP])}** ({pct(p_usd.loc['Maximum drawdown', PP])} in dollars); "
             f"its Sharpe ratio against Euribor was {p_eur.loc['Sharpe ratio', PP]:.2f}, against "
             f"{p_eur.loc['Sharpe ratio', bench]:.2f} for the {bench}."]
    if falling.any() and (~falling).any():
        down, up = dollar_p[falling].mean(), dollar_p[~falling].mean()
        notes.append(f"On average, the dollar {'gained' if down >= 0 else 'lost'} {pct(abs(down))} against the euro "
                     f"in the months in which US equities fell, and {'gained' if up >= 0 else 'lost'} "
                     f"{pct(abs(up))} in those in which they rose.")
    if crises_eur is not None:
        n_crises = len(crises_eur)
        rose = int((crises_eur['Dollar against the euro'] > 0).sum())
        better = int((crises_eur[PP] > crisis_usd[PP]).sum())
        gained = int((crises_eur[PP] > 0).sum())
        notes.append(f"The dollar rose against the euro in "
                     f"{f'all {n_crises}' if rose == n_crises else f'{rose} of the {n_crises}'} equity crises of the "
                     f"period, and the Permanent Portfolio did better in euros than in dollars in "
                     f"{'all of them' if better == n_crises else f'{better} of them'}"
                     + (f", gaining value in euros from the equity peak to the trough in {gained}" if gained else '')
                     + '.')
    with brief:
        st.markdown('#### In brief')
        st.markdown('\n'.join(f'- {s}' for s in notes))
        st.caption('Section 17 of the notebook discusses these results. They depend on the path of the exchange rate '
                   'over the chosen period, and they leave out currency hedging, which would remove most of the '
                   'effect of the exchange rate at a cost or gain set by the difference between dollar and euro '
                   'interest rates.')


def section_method():
    left, right = st.columns([3, 2], gap='large')
    with left:
        st.subheader('Consistency with the notebook')
        checks = []
        for label, (panel, rf), saved_name in (
                ('Table 1 of the notebook', PANELS[('USD', 'Nominal')], 'table1_performance.csv'),
                ('the real-terms Table 1 of Section 16', PANELS[('USD', 'Real')], 'table1_real_terms.csv'),
                ('Table 1 in euros of Section 17', PANELS[('EUR', 'Nominal')], 'table1_euros.csv'),
                ('Table 1 in real euros of Section 17', PANELS[('EUR', 'Real')], 'table1_euros_real.csv')):
            res_d, _, _ = an.simulate(panel, rf=rf)
            table_d, _ = an.performance_tables(res_d)
            reference = notebook_table(saved_name)
            if 'Growth of €1' in reference.index:            # Section 17 labels the last row in euros
                table_d = table_d.rename(index={'Growth of $1': 'Growth of €1'})
            checks.append((label, table_d - reference, reference.notna().sum().sum()))
        for label, diff, n_values in checks:
            gap = diff.abs().max().max()
            if gap < 5e-5:
                st.success(f'With the notebook\'s assumptions, the app reproduces {label} to four decimals (largest '
                           f'difference across the {n_values} reported values: {gap:.1e}).', icon=':material/verified:')
            else:
                st.error(f'The app differs from {label} by up to {gap:.2e}.', icon=':material/error:')
        st.caption('Checked on every load against table1_performance.csv, table1_real_terms.csv, table1_euros.csv and '
                   'table1_euros_real.csv in notebook/outputs. The automated tests in the repository also verify that '
                   'the functions in core.py are identical to the notebook\'s.')
        show_data(checks[0][1], 'Show the differences', file_name='consistency_check.csv')

        st.subheader('Method')
        st.markdown(
            'The portfolio starts with 25% in VTI, TLT, GLD and Treasury bills, after which the weights drift '
            'with relative prices. Under the strategy\'s rule, the weights are checked at every month-end and, '
            'if any lies outside the 15–35% band, all four are reset to 25% at that month\'s closing prices, so '
            'that the new weights earn the next month\'s return and no decision uses information that was not '
            'yet available. Costs are charged on every dollar bought and sold; the initial purchase and taxes '
            'are not charged, and fund expenses are already reflected in the prices.\n\n'
            'Returns are monthly simple total returns from adjusted closing prices (Yahoo Finance). Cash earns '
            'the three-month Treasury-bill rate (FRED, TB3MS), divided by 12 and lagged one month, which is also '
            'the risk-free rate. The benchmark is a constant mix of VTI and US aggregate bonds (AGG), rebalanced '
            'monthly. Means are annualized by multiplying by 12 and volatilities by √12.\n\n'
            'With *Real* selected at the top of the page, every monthly return, Treasury bills included, is '
            'deflated by US consumer-price inflation, (1 + r) / (1 + π) − 1, where π is the monthly change in the '
            'consumer price index for all urban consumers before seasonal adjustment (FRED, CPIAUCNS), and the '
            'same functions are applied to the deflated returns, as in Section 16 of the notebook. Deflation leaves '
            'every rebalancing decision unchanged, values are expressed at the prices of the month in which the '
            'money is invested, and the Sharpe ratio and alpha are measured against the real return of Treasury '
            'bills. The index for October 2025, which the Bureau of Labor Statistics did not publish because prices '
            'were not collected during the federal shutdown, is the geometric mean of September and November.\n\n'
            'With *EUR* selected, the same portfolios are measured by an investor in the euro area who holds the '
            'dollar funds without hedging the currency, as in Section 17 of the notebook. Every monthly return, '
            'Treasury bills included, is converted at month-end exchange rates (FRED, DEXUSEU: noon buying rates in '
            'New York), (1 + r)(1 + e) − 1, where e is the monthly change in the value of one dollar in euros; the '
            'conversion leaves every rebalancing decision unchanged. The risk-free rate of the Sharpe ratio and '
            'alpha is the three-month Euribor (European Central Bank), divided by 12 and lagged one month like the '
            'Treasury-bill rate, and *Real* deflates returns in euros by euro-area inflation, measured by the '
            'harmonised index of consumer prices (FRED, CP0000EZCCM086NEST, from Eurostat). Crises keep the dates '
            'of Section 14, determined by US equities in dollars, in every currency and terms.\n\n'
            'The assumptions in the sidebar are passed to the same functions the notebook uses. Changing the '
            'period restarts the strategy from equal weights in the first month chosen, which differs from the '
            'sub-period estimates in Section 15 of the notebook, where the full-sample backtest is evaluated '
            'over shorter windows. The overview sentences, the sleeve leaders and the crisis panels present '
            'results the notebook reports in its tables, the Inflation tab presents the analysis of Section 16 '
            'for the chosen period, currency and inflation thresholds, and the Currency tab that of Section 17 for '
            'the chosen period.')
    with right:
        st.subheader('Data')
        sample = f'{month_label(MONTHS[0])} to {month_label(MONTHS[-1])}'
        st.table(pd.DataFrame({
            'Series': ['VTI, TLT, GLD, AGG', 'TB3MS', 'CPIAUCNS (CPI-U)', 'DEXUSEU (dollars per euro)',
                       'Euribor, three months', 'HICP, euro area'],
            'Source': ['Yahoo Finance (yfinance)', 'FRED, St. Louis Fed', 'FRED (Bureau of Labor Statistics)',
                       'FRED (Federal Reserve Board)', 'European Central Bank', 'FRED (Eurostat)'],
            'Months': [sample, sample, f'{cpi.index[0]:%b %Y} to {cpi.index[-1]:%b %Y}',
                       f'{eurusd.index[0]:%b %Y} to {eurusd.index[-1]:%b %Y}', sample,
                       f'{hicp.index[0]:%b %Y} to {hicp.index[-1]:%b %Y}'],
            'Observations': [len(rets), len(rets), len(cpi), len(eurusd), len(rf_eur), len(hicp)]}
        ).set_index('Series'), alt='Data sources')
        st.caption('Returns cover the months of the sample; the price indices start twelve months earlier, so that '
                   'inflation over the previous twelve months is defined from the first month, and the exchange rate '
                   'at the month-end before the first return. Interest rates are the previous month\'s averages. Live '
                   'data are downloaded once per server session. If a source does not respond, the app uses the data '
                   'saved by the notebook\'s own run, and says so at the top of the page.')
        show_data(rets_p, 'Show the monthly returns', file_name='monthly_returns.csv')
        show_data(pd.DataFrame({'CPI': cpi, 'Monthly inflation': cpi.pct_change(fill_method=None),
                                '12-month inflation': cpi.pct_change(12, fill_method=None)}),
                  'Show the consumer price index', file_name='consumer_prices.csv')
        show_data(pd.DataFrame({'Dollars per euro': eurusd, 'Dollar return in euros': dollar_return,
                                'Euribor return': rf_eur, 'HICP': hicp,
                                'Monthly euro-area inflation': hicp.pct_change(fill_method=None)}),
                  'Show the exchange rate, Euribor and the HICP', file_name='euro_data.csv')
        show_data(results, 'Show the simulation results', file_name='simulation_results.csv')

        if 'OWNER/REPO' not in REPO_URL:
            st.subheader('Source')
            with st.container(horizontal=True):
                st.link_button('GitHub repository', REPO_URL, icon=':material/code:')
                st.link_button('Notebook', f'{REPO_URL}/blob/main/{NOTEBOOK_PATH}', icon=':material/description:')
                st.link_button('Open in Colab', f'https://colab.research.google.com/github/'
                               f'{REPO_URL.split("github.com/")[1]}/blob/main/{NOTEBOOK_PATH}',
                               icon=':material/play_circle:')


# ----------------------------------------------------------------------------------------------------------
# Views: the client view tells the story; the analyst view has every analysis of the notebook
# ----------------------------------------------------------------------------------------------------------

if view == 'Client':
    sections = {'Overview': section_overview, 'Crises': section_crises, 'Four seasons': section_regimes,
                'Inflation': section_inflation, 'Currency': section_currency, 'Method and data': section_method}
else:
    sections = {'Overview': section_overview, 'Performance': section_performance, 'Risk': section_risk,
                'Rebalancing': section_rebalancing, 'Crises': section_crises,
                'Rules': section_rules, 'Calendar years': section_calendar,
                'Four seasons': section_regimes, 'Inflation': section_inflation, 'Currency': section_currency,
                'Method and data': section_method}
tabs = st.tabs(list(sections), key=f'tab_{view.lower()}', bind='query-params')
for tab, render in zip(tabs, sections.values()):
    with tab:
        render()

st.divider()
st.caption('Educational project for 2270 Financial Modeling, Nova SBE. Results are historical simulations, are not '
           'a forecast of future returns and do not constitute investment advice.')
