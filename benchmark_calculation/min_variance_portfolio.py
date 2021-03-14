from datetime import datetime
from datetime import timedelta
import pandas as pd
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from buy_and_hold import get_ticker_to_closing_prices
from common import *

START_DATE = datetime(2006, 1, 5)
INITIAL_PORTFOLIO_VAL = 100000

# Not sure if we want to compute metrics using test period
# if so, use the following:


def construct_price_df():
    dates, ticker_to_closing_prices = get_ticker_to_closing_prices(
        TEST_START_DATE, TEST_END_DATE
    )

    tickers = ticker_to_closing_prices.keys()
    prices = list(zip(*[ticker_to_closing_prices[ticker] for ticker in tickers]))
    df = pd.DataFrame(
        prices,
        index=dates,
        columns=tickers,
    )

    mu = mean_historical_return(df)
    S = CovarianceShrinkage(df).ledoit_wolf()

    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    ef.portfolio_performance(verbose=True)
    # Expected annual return: 31.9%, Sharpe Ratio: 3.15


def get_mean_variance_share_allocation():
    dates, ticker_to_closing_prices = get_ticker_to_closing_prices(
        START_DATE, TEST_START_DATE - timedelta(days=1)
    )

    tickers = ticker_to_closing_prices.keys()
    prices = list(zip(*[ticker_to_closing_prices[ticker] for ticker in tickers]))
    df = pd.DataFrame(
        prices,
        index=dates,
        columns=tickers,
    )

    mu = mean_historical_return(df)
    S = CovarianceShrinkage(df).ledoit_wolf()

    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()

    _, ticker_to_closing_prices_at_test_start = get_ticker_to_closing_prices(
        TEST_START_DATE, TEST_END_DATE
    )
    prices_at_test_start = pd.Series(
        [float(ticker_to_closing_prices_at_test_start[ticker][0]) for ticker in tickers],
        index=tickers
    )
    da = DiscreteAllocation(
        weights, prices_at_test_start, total_portfolio_value=INITIAL_PORTFOLIO_VAL
    )
    allocation, leftover = da.lp_portfolio()
    for ticker in tickers:
        if ticker not in allocation:
            allocation[ticker] = 0

    return allocation


def main():
    date_to_daily_risk_free_rate = get_date_to_daily_risk_free_rates(
        TEST_START_DATE, TEST_END_DATE
    )
    dates, ticker_to_closing_prices = get_ticker_to_closing_prices(
        TEST_START_DATE, TEST_END_DATE
    )
    daily_risk_free_rates = fill_missing_daily_rf_rates(
        date_to_daily_risk_free_rate, dates
    )

    mv_allocations = get_mean_variance_share_allocation()
    # Compute metrics
    mv_annualized_return, mv_sharpe = compute_metrics(
        mv_allocations, ticker_to_closing_prices, daily_risk_free_rates
    )
    mv_annualized_return = round(mv_annualized_return, 2)
    mv_sharpe = round(mv_sharpe, 2)
    print(f"Mean Variance strategy - Annualized Expected return: {mv_annualized_return}, Sharpe Ratio: {mv_sharpe}")
    # Mean Variance strategy - Annualized Expected return: 0.13, Sharpe Ratio: 1.11


if __name__ == "__main__":
    main()
