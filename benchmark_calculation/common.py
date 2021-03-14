import csv
import os
from datetime import datetime
import numpy as np

# test start/end dates may need adjustment
# once we get more specific dates
# currently based on trading days in 2006/01/01 - 2017/12/31
TEST_START_DATE = datetime(2016, 3, 15)
TEST_END_DATE = datetime(2017, 12, 29)
CSVS_REL_PATH = "../data_preprocessing/csvs"
CSV_SUFFIX = "_technical_data.csv"
CSV_DATE_FMT = "%Y-%m-%d %H:%M:%S"
TRADING_DAYS_PER_YEAR = 252


def get_date_to_daily_risk_free_rates(start_date, end_date):
    """
    Returns a dictionary of date to daily risk free rate
    from the start date to the end date (in order)
    """
    date_to_daily_risk_free_rate = {}
    with open(
        "../data_preprocessing/csvs/US_treasury_daily_risk_free_rates.csv", "r"
    ) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            date = datetime.strptime(row[0], CSV_DATE_FMT)
            if date >= start_date and date <= end_date:
                date_to_daily_risk_free_rate[date] = float(row[1])
    return date_to_daily_risk_free_rate


def fill_missing_daily_rf_rates(date_to_daily_rf_rate, required_dates):
    """
    Given a mapping from date to daily risk free rates and
    a list of days for which we need to know the daily risk free rates,
    fill out any missing days from the mapping.

    This is required because there are some days when the stock market
    was open but the US Treasury did not publish yield curve rates.
    Approximate missing data using the previous day when rates were published.
    """
    daily_risk_free_rates = []
    for date in required_dates:
        if date not in date_to_daily_rf_rate:
            missing_date_index = required_dates.index(date)
            # Assume the daily rate exists for the day before
            date_to_daily_rf_rate[date] = date_to_daily_rf_rate[required_dates[missing_date_index - 1]]
        daily_risk_free_rates.append(date_to_daily_rf_rate[date])
    return daily_risk_free_rates


def get_ticker_to_closing_prices(start_date, end_date):
    """
    Returns a dictionary where keys are stock tickers and
    the items are a list of the closing price of the stock
    from the start date to the end date (in order)

    Also returns a list of dates corresponding to the closing prices
    """
    csvs = [f"{CSVS_REL_PATH}/{f}" for f in os.listdir(CSVS_REL_PATH) if f.endswith(CSV_SUFFIX)]

    # Find CSV row number of start and end date
    start_date_row_num = 0
    end_date_row_num = 0
    with open(csvs[0], newline="") as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        rows = rows[1:]  # remove header
        start_date_row_num += 1
        end_date_row_num += 1
        for row in rows:
            date = datetime.strptime(row[0], CSV_DATE_FMT)
            if date < start_date:
                start_date_row_num += 1
            if date <= end_date:
                end_date_row_num += 1
            else:
                break

    # Compute closing prices for start and end dates for each stock
    ticker_to_closing_prices = {}
    dates = set()
    for csv_path in csvs:
        ticker = csv_path.lstrip(CSVS_REL_PATH).rstrip(CSV_SUFFIX)
        with open(csv_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader][start_date_row_num:end_date_row_num]
            ticker_to_closing_prices[ticker] = [
                float(row[2]) for row in rows
            ]
            # Only set this the first time
            if not dates:
                dates = set([datetime.strptime(row[0], CSV_DATE_FMT) for row in rows])

    dates = sorted(list(dates))
    return dates, ticker_to_closing_prices


def compute_metrics(
    ticker_to_share_num, ticker_to_closing_prices, daily_rf_rates
):
    """
    Given an allocation of shares, historical closing prices, and
    daily risk free rates, return the annualized expected return
    and sharpe ratio of the portfolio.
    """
    daily_return_rates = []
    num_trading_days = len(list(ticker_to_closing_prices.values())[0])
    for i in range(1, num_trading_days):
        # Compute daily return rate
        daily_total_profit = 0.0
        total_balance_yesterday = 0.0
        for ticker, closing_prices in ticker_to_closing_prices.items():
            price_today = closing_prices[i]
            price_yesterday = closing_prices[i - 1]
            profit = (price_today - price_yesterday) * ticker_to_share_num[ticker]
            daily_total_profit += profit
            total_balance_yesterday += price_yesterday * ticker_to_share_num[ticker]

        daily_return_rate = daily_total_profit / total_balance_yesterday
        daily_return_rates.append(daily_return_rate)

    daily_return_rates = np.array(daily_return_rates)

    # Compute required statistics
    annualized_expected_return = (1 + daily_return_rates.mean()) ** TRADING_DAYS_PER_YEAR - 1
    # See https://quant.stackexchange.com/questions/28385/what-value-should-the-risk-free-monthly-return-rate-be-sharpe-ratio-calculation
    # for scaling sharpe ratio
    sharpe_ratio = (
        np.sqrt(TRADING_DAYS_PER_YEAR) *
        (daily_return_rates - np.array(daily_rf_rates[1:])).mean() / daily_return_rates.std()
    )

    return annualized_expected_return, sharpe_ratio
