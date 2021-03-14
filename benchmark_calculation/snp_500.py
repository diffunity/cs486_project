import csv
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import yfinance as yf


# test start/end dates may need adjustment
# once we get more specific dates
# currently based on trading days in 2006/01/01 - 2017/12/31
TEST_START_DATE = datetime(2016, 3, 15)
TEST_END_DATE = datetime(2017, 12, 29)
SNP_500_TICKER = "^GSPC"
CSVS_REL_PATH = "../data_preprocessing/csvs"
CSV_DATE_FMT = "%Y-%m-%d %H:%M:%S"
TRADING_DAYS_PER_YEAR = 252


def get_date_to_daily_risk_free_rates():
    """
    Returns a dictionary of date to daily risk free rate
    from the start date to the end date (in order)
    """
    date_to_daily_risk_free_rate = {}
    with open(
        f"{CSVS_REL_PATH}/US_treasury_daily_risk_free_rates.csv", "r"
    ) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            date = datetime.strptime(row[0], CSV_DATE_FMT)
            if date >= TEST_START_DATE and date <= TEST_END_DATE:
                date_to_daily_risk_free_rate[date] = float(row[1])
    return date_to_daily_risk_free_rate


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
    print(snp_500_closing_prices)
    print(snp_500_daily_return_rates)
    annualized_expected_return = (1 + snp_500_daily_return_rates.mean()) ** TRADING_DAYS_PER_YEAR - 1
    sharpe_ratio = (
        np.sqrt(TRADING_DAYS_PER_YEAR) *
        (snp_500_daily_return_rates - np.array(daily_rf_rates[1:])).mean() / snp_500_daily_return_rates.std()
    )
    return annualized_expected_return, sharpe_ratio


def main():
    date_to_daily_risk_free_rate = get_date_to_daily_risk_free_rates()
    date_to_snp_500_prices = get_date_to_snp_500_prices()
    print(date_to_snp_500_prices)

    # There were some days when the stock market was open but the US
    # Treasury did not publish yield curve rates
    # For such dates, approximate using the previous day when
    # rates were published
    daily_risk_free_rates = []
    dates = list(date_to_snp_500_prices.keys())
    for date in dates:
        if date not in date_to_daily_risk_free_rate:
            missing_date_index = dates.index(date)
            date_to_daily_risk_free_rate[date] = date_to_daily_risk_free_rate[dates[missing_date_index - 1]]
        daily_risk_free_rates.append(date_to_daily_risk_free_rate[date])

    snp_500_prices = list(date_to_snp_500_prices.values())
    snp_annualized_return, snp_sharpe = get_snp_500_metrics(snp_500_prices, daily_risk_free_rates)

    print(f"S&P 500 index - Annualized Expected return: {snp_annualized_return}, Sharpe Ratio: {snp_sharpe}")
    # S&P 500 index - Annualized Expected return: 0.17289162879698505, Sharpe Ratio: 1.7899473816048437


if __name__ == "__main__":
    main()
