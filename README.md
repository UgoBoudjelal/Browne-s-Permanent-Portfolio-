# Harry Browne's Permanent Portfolio: backtest and interactive explorer

Group G · 2270 Financial Modeling · Nova SBE

[![tests](https://github.com/UgoBoudjelal/Browne-s-Permanent-Portfolio-/actions/workflows/tests.yml/badge.svg)](https://github.com/UgoBoudjelal/Browne-s-Permanent-Portfolio-/actions/workflows/tests.yml)

**Application:** *(link added after deployment)*
**Notebook:** [`notebook/Group_G_Permanent_Portfolio.ipynb`](notebook/Group_G_Permanent_Portfolio.ipynb) ·
[Open in Colab](https://colab.research.google.com/github/UgoBoudjelal/Browne-s-Permanent-Portfolio-/blob/main/notebook/Group_G_Permanent_Portfolio.ipynb)

This repository accompanies the group's investor prospectus. The notebook backtests the Permanent Portfolio proposed by Browne (1999), which holds 25% each in US equities (VTI), long-term US Treasuries (TLT), gold (GLD) and Treasury bills, and restores equal weights whenever a sleeve leaves a 15–35% tolerance band at a month-end review. Over January 2005 to October 2026 the strategy is evaluated net of transaction costs against a 60/40 portfolio of VTI and US aggregate bonds (AGG) and against an all-equity portfolio. The notebook runs end to end in a fresh Google Colab session and retrieves all data at run time from Yahoo Finance and FRED.

The Streamlit application lets an investor change the assumptions of the backtest and observe their effect on performance and on the comparison with the benchmark. The rebalancing rule (tolerance band, calendar reset or buy and hold), the width of the band, the review frequency, the transaction cost, the equity share of the benchmark, the investment period and the amount invested can all be varied. The results are organised as in the notebook: cumulative performance and the full set of performance measures, drawdowns and risk, rebalancing activity and costs, behaviour in five equity crises, the comparison of rebalancing rules and calendar-year returns. The settings are kept in the page address, so a scenario can be shared as a link.

## Running the application locally

```
pip install -r requirements.txt
streamlit run app.py
```

The application was tested with Python 3.12. The first load downloads the data, which takes a few seconds.

## Consistency with the notebook

The application computes every result with the notebook's own code. `core.py` contains exact copies of the functions defined in the notebook (data retrieval, the backtesting engine and the performance measures), and `analysis.py` repeats the code of the notebook cells that use them, with the model parameters as arguments. The tests in `tests/` verify that each function in `core.py` is identical, character for character, to its definition in the notebook; that with the notebook's assumptions `analysis.py` reproduces every table the notebook saves in `notebook/outputs/`; and that the application runs under every combination of rebalancing rule and review frequency and reproduces Table 1 with its default settings. The application also performs this last comparison each time it loads and reports the result in its *Method and data* tab. The tests run without network access and are executed on every push:

```
pip install -r requirements-dev.txt
python -m pytest -q tests
```

## Data

Monthly prices adjusted for splits and distributions are downloaded from Yahoo Finance through `yfinance`, and the three-month Treasury-bill rate (TB3MS) from FRED, in a single request each per server session. If either source does not respond, for instance because of rate limiting, the application uses the monthly returns saved by the notebook's own run in `notebook/outputs/returns_monthly.csv`, which cover the same months, and states this at the top of the page. The notebook itself never reads saved data.

## Repository structure

```
app.py                  Streamlit application
core.py                 exact copies of the notebook's functions
analysis.py             the notebook's result cells as functions of the assumptions
notebook/               the notebook and the tables it saves (outputs/)
tests/                  consistency tests (notebook vs core.py, analysis.py and the application)
requirements.txt        packages used by the application, pinned to the tested versions
.streamlit/config.toml  light and dark themes
```

## Availability

The application is hosted on Streamlit Community Cloud, which puts applications to sleep after a period without visits. If the page reports that the application is asleep, selecting *Yes, get this app back up* restarts it in about a minute.

## Reference

Browne, H. (1999). *Fail-safe investing: Lifelong financial security in 30 minutes*. St. Martin's Press.

Results are historical simulations for an educational project; they are not a forecast of future returns and do not constitute investment advice.
