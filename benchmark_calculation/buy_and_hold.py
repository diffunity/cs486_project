import os
import csv
from datetime import datetime
from datetime import timedelta
import numpy as np
from common import *

CAPITAL_PER_STOCK = 10000  # USD


def get_buy_and_hold_share_allocation(ticker_to_closing_prices):
    """
    Compute the initial number of stocks purchased using the
    buy and hold strategy.
    """
    # Compute how many shares we have for each stock
    # NOTE: may want to change this so that remaining balance is used to
    #       buy up cheapest stock? (to minimize amount of unutilized capital)
    ticker_to_share_num = {}
    for ticker, closing_prices in ticker_to_closing_prices.items():
        initial_price = closing_prices[0]
        num_purchased = CAPITAL_PER_STOCK // initial_price
        ticker_to_share_num[ticker] = num_purchased

    return ticker_to_share_num


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

    bnh_allocations = get_buy_and_hold_share_allocation(ticker_to_closing_prices)
    # Compute metrics
    bnh_annualized_return, bnh_sharpe = compute_metrics(
        bnh_allocations, ticker_to_closing_prices, daily_risk_free_rates
    )
    bnh_annualized_return = round(bnh_annualized_return, 2)
    bnh_sharpe = round(bnh_sharpe, 2)
    print(f"Buy and hold strategy - Annualized Expected return: {bnh_annualized_return}, Sharpe Ratio: {bnh_sharpe}")
    # Buy and hold strategy - Annualized Expected return: 0.24, Sharpe Ratio: 1.96


if __name__ == "__main__":
    main()
