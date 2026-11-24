"""Permanent Portfolio explorer: the Streamlit application of the Group G notebook.

Run locally with `streamlit run app.py`. Every number is computed with core.py (exact copies of the
notebook's functions) and analysis.py (the notebook's result cells as functions of the assumptions);
tests/ verifies that the default settings reproduce the notebook's tables, in nominal and in real terms.
"""
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import analysis as an
from core import download_cpi, format_table, load_data, real_returns

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

@st.cache_data(show_spinner='Downloading monthly prices (Yahoo Finance), the T-bill rate and consumer prices '
                             '(FRED)...')
def live_data():
    _, rets = load_data(an.START, an.END, an.TICKERS)
    cpi = download_cpi(an.CPI_START, an.END)
    return rets, cpi


@st.cache_data(show_spinner=False)
def saved_data():
    rets = pd.read_csv(OUTPUTS / 'returns_monthly.csv', index_col=0, parse_dates=True)
    cpi = pd.read_csv(OUTPUTS / 'consumer_prices.csv', index_col=0, parse_dates=True)['CPI']
    return rets, cpi


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
def rule_table(rets_p, cost, w_bench):
    # Section 15 for the chosen period, cost and benchmark
    results, _, _ = an.simulate(rets_p, cost=cost, w_stocks_bench=w_bench)
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


def time_chart(wide, fmt, y_title, key, colors, log=False, under=(), over=(), height=340, y_domain=None, dashed=(),
               legend=True):
    """Lines over time with a hover crosshair; the tooltip lists every series at the hovered month."""
    names = list(wide.columns)
    data = wide.rename_axis('Date').reset_index()
    long = data.melt('Date', var_name='Series', value_name='Value')
    safe = data.copy()
    safe.columns = ['Date'] + [f's{i}' for i in range(len(names))]

    color = alt.Color('Series:N', scale=alt.Scale(domain=names, range=colors[:len(names)]),
                      legend=alt.Legend(title=None, orient='top', direction='horizontal') if legend else None)
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
        encoding['strokeDash'] = alt.StrokeDash('Series:N', legend=None, scale=alt.Scale(
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

rets, cpi, source = get_data()
MONTHS = list(rets.index.strftime('%Y-%m'))

# Section 16: inflation and the return panel in real terms (every series deflated by the month's inflation)
inflation, inflation_12m = an.inflation_rates(cpi, rets.index)
real_rets = real_returns(rets, inflation)


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
    amount = st.number_input('Amount invested (USD)', min_value=100, max_value=100_000_000,
                             value=DEFAULTS['amount'], step=1_000, key='amount', bind='query-params')
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
        st.badge('Live data: Yahoo Finance and FRED', icon=':material/cloud_done:', color='green')
    else:
        st.badge("Data saved by the notebook's run", icon=':material/save:', color='orange')
    if is_default:
        st.badge("Notebook's assumptions", icon=':material/verified:', color='primary')
    else:
        st.badge('Custom assumptions', icon=':material/tune:', color='violet')
    st.space('stretch')
    terms = st.segmented_control('Returns', ['Nominal', 'Real'], default='Nominal', required=True, key='terms',
                                 bind='query-params', label_visibility='collapsed',
                                 help='Nominal: returns in US dollars. Real: returns after US consumer-price '
                                      'inflation, with values at the prices of the month in which the money is '
                                      'invested.')
    view = st.segmented_control('View', ['Client', 'Analyst'], default='Client', required=True, key='view',
                                bind='query-params', label_visibility='collapsed',
                                help='Client: the key results in plain language. Analyst: every table and '
                                     'analysis of the notebook.')
REAL = terms == 'Real'
st.caption(f'Rule: {rule_text} · cost {cost:.2%} per side · benchmark {bench} VTI/AGG · '
           f'{month_label(first)} to {month_label(last)}' + (' · returns after inflation' if REAL else ''))

if st.session_state.get('data_error') and source == 'saved':
    with st.container(border=True):
        st.warning('Yahoo Finance or FRED did not respond, so the page uses the monthly returns and consumer prices '
                   "saved by the notebook's own run, which cover the same months. Live data can be retried in a "
                   'minute.', title='Data source busy, try again', icon=':material/cloud_off:')

        def retry():
            st.session_state['use_saved_data'] = False
            st.session_state['data_error'] = None
            live_data.clear()

        st.button('Try live data again', on_click=retry, icon=':material/refresh:', key='retry')

nominal_p, real_p = rets.loc[first:last], real_rets.loc[first:last]
rets_p = real_p if REAL else nominal_p            # every section below follows the choice of returns
if len(rets_p) < 36:
    st.warning('Choose a period of at least 36 months: shorter samples give unreliable estimates of '
               'volatility, beta and alpha.', icon=':material/date_range:')
    st.stop()

# ----------------------------------------------------------------------------------------------------------
# Model: the notebook's calculations under the chosen assumptions
# ----------------------------------------------------------------------------------------------------------

# Both runs are kept: the chosen one drives every section, and the Inflation tab compares the two
runs = {t: an.simulate(p, low, high, check, cost, cost_stress, w_bench) for t, p in
        (('Nominal', nominal_p), ('Real', real_p))}
results, weights_after, weights_before = runs[terms]
results_nom, results_real = runs['Nominal'][0], runs['Real'][0]
perf_table, perf_vs_stocks = an.performance_tables(results, bench)
perf_real = perf_table if REAL else an.performance_tables(results_real, bench)[0]
levels = sorted(set(an.COST_LEVELS + [cost, cost_stress]))
with np.errstate(divide='ignore'):            # without trades the break-even cost is undefined
    cost_summary, sensitivity = an.cost_analysis(results, cost, cost_stress, levels)
record = an.rebalancing_record(results, weights_before, cost, cost_stress)
crisis_table, recovery_table = an.crisis_analysis(results, bench)
rebalance_dates = list(results.index[results['turnover'] > 0])

colors3 = P['series']
names3 = [PP, bench, VTI]
growth = pd.DataFrame({PP: an.wealth_index(results['pp_net']), bench: an.wealth_index(results['bench_6040']),
                       VTI: an.wealth_index(results['stocks'])})          # value of $1, from 1 the month before
wealth = growth * amount
start_ts = rets_p.index[0] - pd.DateOffset(months=1)        # the month-end at which the money is invested
start_label, end_label = f'{start_ts:%B %Y}', f'{rets_p.index[-1]:%B %Y}'
REAL_NOTE = ' All returns are after US consumer-price inflation.' if REAL else ''

# Consumer prices over the chosen period (Section 16)
cpi_p = cpi.loc[start_ts:rets_p.index[-1]]
price_rise = cpi_p.iloc[-1] / cpi_p.iloc[0] - 1
inflation_rate = (1 + price_rise) ** (an.PERIODS / len(rets_p)) - 1
inflation_p, inflation_12m_p = inflation.loc[first:last], inflation_12m.loc[first:last]

# KPI row: the Permanent Portfolio net of costs, with the difference from the frictionless benchmark
pp_m, b_m = perf_table[PP], perf_table[bench]
final_pp, final_b = amount * pp_m['Growth of $1'], amount * b_m['Growth of $1']
diff_final = final_pp - final_b
with st.container(horizontal=True):
    st.metric(f'Real value of ${amount:,.0f}' if REAL else f'Value of ${amount:,.0f}', f'${final_pp:,.0f}',
              f"{'+' if diff_final >= 0 else '-'}${abs(diff_final):,.0f}", delta_description=f'vs {bench}',
              help=f'At {start_label} prices' if REAL else None, border=True)
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
if REAL:
    st.caption(f'After inflation: every return is deflated by US consumer prices (CPI-U), and values are at '
               f'{start_label} prices.')


# ----------------------------------------------------------------------------------------------------------
# Sections (each tab is one function, shared by the client and analyst views)
# ----------------------------------------------------------------------------------------------------------

def section_overview():
    # The key results in plain sentences, written from the computed numbers
    v_m = perf_table[VTI]
    lines = [
        f"From the end of {start_label} to the end of {end_label}, ${amount:,.0f} in the Permanent Portfolio grew to "
        f"**${final_pp:,.0f}**{f' at {start_label} prices' if REAL else ''}, a {'real ' if REAL else ''}compound "
        f"return of {pct(pp_m['Compound annual return'])} a year, against "
        f"${final_b:,.0f} ({pct(b_m['Compound annual return'])}) for the {bench} and "
        f"${amount * v_m['Growth of $1']:,.0f} ({pct(v_m['Compound annual return'])}) for an all-equity portfolio.",
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
        lines.append(f"US consumer prices rose by {pct(price_rise, 1)} over the period, {pct(inflation_rate)} a year; "
                     f"every value on this page is after inflation, at {start_label} prices.")
    else:
        lines.append(f"US consumer prices rose by {pct(price_rise, 1)} over the period, {pct(inflation_rate)} a year, "
                     f"so after inflation the Permanent Portfolio grew by "
                     f"{pct(perf_real.loc['Compound annual return', PP])} a year, against "
                     f"{pct(perf_real.loc['Compound annual return', bench])} for the {bench}. Select *Real* at the "
                     "top of the page to see every result after inflation.")
    with st.container(border=True):
        st.markdown('#### In brief')
        st.markdown('\n'.join(f'- {s}' for s in lines))
        st.caption(f'Permanent Portfolio net of {cost:.2%} costs per side; {bench} before costs. The sentences '
                   'update with the assumptions in the sidebar.' + REAL_NOTE)

    st.subheader(f'{"Real value" if REAL else "Value"} of ${amount:,.0f} invested at the end of {start_label}')
    with st.container(horizontal=True, vertical_alignment='center'):
        st.space('stretch')
        log_scale = st.toggle('Log scale', key='log_scale')
        show_marks = st.toggle('Rebalancing dates', key='show_marks', value=True)
    shade, labels = crisis_shading(crisis_table)
    under = shade + ([vertical_marks(rebalance_dates)] if show_marks and 0 < len(rebalance_dates) <= 60 else [])
    y_title = f'Value at {start_ts:%b %Y} prices (USD)' if REAL else 'Portfolio value (USD)'
    st.altair_chart(time_chart(wealth, '$,.0f', y_title, 'wealth', colors3, log=log_scale,
                               under=under, over=labels, height=400),
                    alt=f'Growth of ${amount:,.0f} in the Permanent Portfolio, the {bench} benchmark and VTI, with '
                        'the equity crises shaded')
    st.caption('Shaded: equity crises, from the equity peak to the trough. Dashed lines: months in which the '
               'portfolio was rebalanced back to 25% each.' + REAL_NOTE)
    show_data(wealth.round(2), file_name='portfolio_values.csv')

    left, right = st.columns([3, 2], gap='large')
    with left:
        st.markdown('**At a glance**')
        glance = ['Compound annual return', 'Annualized volatility', 'Sharpe ratio', 'Maximum drawdown',
                  'Growth of $1']
        st.table(table_text(format_table(perf_table.loc[glance, names3])), alt='Key measures of the three portfolios')
    with right:
        st.markdown('**What the measures mean**')
        st.markdown(
            '- *Compound return*: the constant annual rate that turns the initial amount into the final value.\n'
            '- *Volatility*: how much monthly returns swing, annualized.\n'
            '- *Sharpe ratio*: return above Treasury bills per unit of volatility.\n'
            '- *Maximum drawdown*: the largest fall from a previous peak in value.\n'
            '- *Growth of $1*: what each dollar invested at the start became.')


def section_performance():
    st.subheader('Performance measures after inflation' if REAL else 'Performance measures')
    st.caption(f'{month_label(first)} to {month_label(last)}. Permanent Portfolio net of {cost:.2%} (base) and '
               f'{cost_stress:.2%} (stress) costs per side; the benchmark without and with base costs. Beta, alpha, '
               f'tracking error and information ratio are measured against the frictionless {bench}; beta is not '
               'annualized.' + (' All returns are after US consumer-price inflation, and the Sharpe ratio and alpha '
                                'are measured against the real return of Treasury bills (Section 16 of the '
                                'notebook).' if REAL else ''))
    st.table(table_text(format_table(perf_table)), alt='Performance measures of the portfolios')
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
                '- *Sharpe ratio*: mean monthly return in excess of T-bills divided by its standard deviation, '
                '× √12.\n'
                '- *Beta*: sensitivity of the excess return to that of the benchmark. *Alpha*: the annualized '
                'excess return that beta does not explain; a *t-statistic* above about 2 in absolute value '
                'indicates significance at 5%.\n'
                '- *Tracking error*: volatility of the return difference from the benchmark. *Information '
                'ratio*: mean return difference divided by tracking error.\n'
                '- *Maximum drawdown*: the largest fall from a previous peak in value.')
    show_data(perf_table, 'Unformatted values', file_name='performance_measures.csv')


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
        st.caption('Standard deviation of the previous 36 monthly returns, annualized.' + REAL_NOTE)
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
        st.caption('The three portfolios in colour; the constituents and the bond index in grey.' + REAL_NOTE)
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
               'rises; the break-even cost is the rate at which it reaches zero.' + REAL_NOTE)


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
               'The crises are dated as in Section 14 of the notebook: the lowest month-end of VTI within a search '
               'window and the highest month-end before it.' +
               (' After inflation, both the dates and the returns are those of real wealth.' if REAL else ''))

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

    left, right = st.columns([3, 2], gap='large')
    with left:
        st.markdown('**Peak-to-trough returns**')
        shown = crisis_table.copy()
        for p in names3:
            shown[p] = [pct(v, 1) for v in crisis_table[p]]
        st.table(shown, alt='Peak-to-trough returns by episode')
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

        st.subheader('Calendar-year real returns of each sleeve' if REAL else 'Calendar-year returns of each sleeve')
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
               'tolerance band of 15–35%, reviewed monthly.' + REAL_NOTE)
    rules_tbl = rule_table(rets_p, cost, w_bench).copy()

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
    st.subheader('Calendar-year real returns' if REAL else 'Calendar-year returns')
    cal_src = pd.DataFrame({PP: results['pp_net'], bench: results['bench_6040'], VTI: results['stocks'],
                            'TLT': rets_p['TLT'], 'GLD': rets_p['GLD'], 'AGG': rets_p['AGG'],
                            'T-bills': rets_p['rf']})
    cal = an.calendar_returns(cal_src)
    cal['Inflation (CPI)'] = an.calendar_returns(inflation_p.to_frame('Inflation'))['Inflation']
    counts = cal_src.groupby(cal_src.index.year).size()
    cal.index = [f'{y}*' if counts[y] < 12 else str(y) for y in cal.index]
    heat = cal.rename_axis('Year').reset_index().melt('Year', var_name='Series', value_name='Return')
    returns_only = heat[heat['Series'] != 'Inflation (CPI)']
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
    # Inflation is shown for reference, on a neutral background rather than on the scale of returns
    rects_cpi = alt.Chart(heat[heat['Series'] == 'Inflation (CPI)']).mark_rect(
        cornerRadius=3, color=P['mid'], stroke=P['soft'], strokeWidth=1).encode(
        x=x, y=y, tooltip=['Year', alt.Tooltip('Return:Q', title='Inflation', format='.1%')])
    text = alt.Chart(heat).mark_text(fontSize=11, color=P['ink']).encode(
        x=x, y=y, text=alt.Text('Return:Q', format='.1%'))
    st.altair_chart(alt.layer(rects, rects_cpi, text).properties(height=24 * len(cal) + 80),
                    alt='Calendar-year returns of the portfolios and of each asset, with the inflation of each year')
    st.caption('Portfolio returns net of costs for the Permanent Portfolio. * Partial year: only the months inside '
               'the chosen period. Blue years are gains, red years losses. Inflation (CPI): the change in US consumer '
               'prices over the same months, for reference.' + REAL_NOTE)
    show_data(cal, file_name='calendar_returns.csv')


def section_inflation():
    # Section 16 of the notebook for the chosen period: purchasing power, real returns and inflation regimes
    r_pp, r_b = perf_real[PP], perf_real[bench]
    kpi = st.columns(4)
    kpi[0].metric('Rise in consumer prices', pct(price_rise, 1), border=True,
                  help=f'US consumer price index (CPI-U) from {start_label} to {end_label}')
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
    values['US consumer prices'] = amount * cpi_p / cpi_p.iloc[0]
    st.altair_chart(time_chart(values, '$,.0f', 'Value (USD)', 'keep', colors3 + [P['muted']],
                               dashed=['US consumer prices'], height=360),
                    alt=f'Nominal value of ${amount:,.0f} in each portfolio against the amount needed to keep pace '
                        'with consumer prices')
    st.caption(f'Value in US dollars of ${amount:,.0f} invested at the end of {start_label}. The dashed line is the '
               'amount that keeps the purchasing power of the initial investment: a portfolio above it has grown in '
               'real terms.')
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
            st.markdown(f'US consumer prices rose by {pct(price_rise, 1)} from {start_label} to {end_label}.')
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
                    alt='US consumer-price inflation over the previous twelve months, with the low and high '
                        'inflation regimes shaded')
    st.caption('Change in US consumer prices (CPI-U) over the twelve months to each month-end. The shaded bands are '
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
    notes = [f"US consumer prices rose by **{pct(price_rise, 1)}** from {start_label} to {end_label}, "
             f"{pct(inflation_rate)} a year: ${amount:,.0f} at the end buys what ${amount / (1 + price_rise):,.0f} "
             f"bought at the start. After inflation, the Permanent Portfolio grew by "
             f"**{pct(r_pp['Compound annual return'])}** a year, against {pct(r_b['Compound annual return'])} for "
             f"the {bench}."]
    high, low_k = labels[2], labels[0]
    if months[high] > 0:
        t = regime_table[high]
        positive = [s for s in SLEEVES if t[s] > 0]
        best = t[SLEEVES].idxmax()
        lead = (f"{SHORT[positive[0]].capitalize()} was the only sleeve with a positive real return "
                f"({pct(t[positive[0]], 1)} a year)" if len(positive) == 1 else
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
        notes.append('Treasury bills lost purchasing power in every inflation regime.')
    with brief:
        st.markdown('#### In brief')
        st.markdown('\n'.join(f'- {s}' for s in notes))
        st.caption(f'Inflation regimes: {threshold(lo)} and {threshold(hi)} over the previous twelve months. '
                   'Section 16 of the notebook discusses these results; Erb and Harvey (2013) find that gold has been '
                   'an unreliable inflation hedge over the horizons that matter to most investors.')

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


def section_method():
    left, right = st.columns([3, 2], gap='large')
    with left:
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
            'The assumptions in the sidebar are passed to the same functions the notebook uses. Changing the '
            'period restarts the strategy from equal weights in the first month chosen, which differs from the '
            'sub-period estimates in Section 15 of the notebook, where the full-sample backtest is evaluated '
            'over shorter windows. The overview sentences, the sleeve leaders and the crisis panels present '
            'results the notebook reports in its tables, and the Inflation tab presents the analysis of Section 16 '
            'for the chosen period and inflation thresholds.')

        st.subheader('Consistency with the notebook')
        checks = []
        for label, panel, saved_name in (('Table 1', rets, 'table1_performance.csv'),
                                         ('the real-terms Table 1 of Section 16', real_rets, 'table1_real_terms.csv')):
            res_d, _, _ = an.simulate(panel)
            table_d, _ = an.performance_tables(res_d)
            reference = notebook_table(saved_name)
            checks.append((label, table_d - reference, reference.notna().sum().sum()))
        for label, diff, n_values in checks:
            gap = diff.abs().max().max()
            if gap < 5e-5:
                st.success(f'With the notebook\'s assumptions the app reproduces {label} of the notebook to four '
                           f'decimals (largest difference across the {n_values} reported values: {gap:.1e}).',
                           icon=':material/verified:')
            else:
                st.error(f'The app differs from {label} of the notebook by up to {gap:.2e}.', icon=':material/error:')
        st.caption('Checked on every load against notebook/outputs/table1_performance.csv and '
                   'table1_real_terms.csv. The automated tests in the repository also verify that the functions in '
                   'core.py are identical to the notebook\'s.')
        show_data(checks[0][1], 'Show the differences', file_name='consistency_check.csv')
    with right:
        st.subheader('Data')
        st.table(pd.DataFrame({
            'Series': ['VTI, TLT, GLD, AGG', 'TB3MS', 'CPIAUCNS (CPI-U)'],
            'Source': ['Yahoo Finance (yfinance)', 'FRED, St. Louis Fed', 'FRED (Bureau of Labor Statistics)'],
            'Months': [f'{month_label(MONTHS[0])} to {month_label(MONTHS[-1])}'] * 2
                      + [f'{cpi.index[0]:%b %Y} to {cpi.index[-1]:%b %Y}'],
            'Observations': [len(rets)] * 2 + [len(cpi)]}).set_index('Series'), alt='Data sources')
        st.caption('Returns cover the months of the sample; the price index starts twelve months earlier, so that '
                   'inflation over the previous twelve months is defined from the first month. Live data are '
                   'downloaded once per server session. If a source does not respond, the app uses the data saved '
                   'by the notebook\'s own run, and says so at the top of the page.')
        show_data(rets_p, 'Show the monthly returns', file_name='monthly_returns.csv')
        show_data(pd.DataFrame({'CPI': cpi, 'Monthly inflation': cpi.pct_change(fill_method=None),
                                '12-month inflation': cpi.pct_change(12, fill_method=None)}),
                  'Show the consumer price index', file_name='consumer_prices.csv')
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
                'Inflation': section_inflation, 'Method and data': section_method}
else:
    sections = {'Overview': section_overview, 'Performance': section_performance, 'Risk': section_risk,
                'Rebalancing and costs': section_rebalancing, 'Crises': section_crises,
                'Rebalancing rules': section_rules, 'Calendar years': section_calendar,
                'Four seasons': section_regimes, 'Inflation': section_inflation, 'Method and data': section_method}
tabs = st.tabs(list(sections), key=f'tab_{view.lower()}', bind='query-params')
for tab, render in zip(tabs, sections.values()):
    with tab:
        render()

st.divider()
st.caption('Educational project for 2270 Financial Modeling, Nova SBE. Results are historical simulations, are not '
           'a forecast of future returns and do not constitute investment advice.')
