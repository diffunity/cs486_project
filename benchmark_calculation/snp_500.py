import csv
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from common import *


SNP_500_TICKER = "^GSPC"


def get_date_to_snp_500_prices():
    date_to_price = {}
    data = yf.download(
        SNP_500_TICKER,
        start=TEST_START_DATE,
        end=TEST_END_DATE + timedelta(days=1),
        interval="1d",
        auto_adjust=True,
    )
    for ts, row in data.iterrows():
        date = datetime(year=ts.year, month=ts.month, day=ts.day)
        date_to_price[date] = float(row["Close"])
    return date_to_price


def get_snp_500_metrics(snp_500_closing_prices, daily_rf_rates):
    snp_500_daily_return_rates = pd.Series(snp_500_closing_prices).pct_change()[1:]
    annualized_expected_return = (1 + snp_500_daily_return_rates.mean()) ** TRADING_DAYS_PER_YEAR - 1
    sharpe_ratio = (
        np.sqrt(TRADING_DAYS_PER_YEAR) *
        (snp_500_daily_return_rates - np.array(daily_rf_rates[1:])).mean() / snp_500_daily_return_rates.std()
    )
    return annualized_expected_return, sharpe_ratio


def main():
    date_to_daily_risk_free_rate = get_date_to_daily_risk_free_rates(
        TEST_START_DATE, TEST_END_DATE
    )
    date_to_snp_500_prices = get_date_to_snp_500_prices()
    dates = list(date_to_snp_500_prices.keys())
    daily_risk_free_rates = fill_missing_daily_rf_rates(
        date_to_daily_risk_free_rate, dates
    )

    snp_500_prices = list(date_to_snp_500_prices.values())
    snp_annualized_return, snp_sharpe = get_snp_500_metrics(snp_500_prices, daily_risk_free_rates)

    snp_annualized_return = round(snp_annualized_return, 2)
    snp_sharpe = round(snp_sharpe, 2)

    print(f"S&P 500 index - Annualized Expected return: {snp_annualized_return}, Sharpe Ratio: {snp_sharpe}")
    # S&P 500 index - S&P 500 index - Annualized Expected return: 0.17, Sharpe Ratio: 1.79


if __name__ == "__main__":
    main()
