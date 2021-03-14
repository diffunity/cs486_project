import os
import csv
from datetime import datetime
from datetime import timedelta
import numpy as np

# test start/end dates may need adjustment
# once we get more specific dates
# currently based on trading days in 2006/01/01 - 2017/12/31
TEST_START_DATE = datetime(2016, 3, 15)
TEST_END_DATE = datetime(2017, 12, 29)
CSVS_REL_PATH = "../data_preprocessing/csvs"
CSV_SUFFIX = "_technical_data.csv"
CSV_DATE_FMT = "%Y-%m-%d %H:%M:%S"
CAPITAL_PER_STOCK = 10000  # USD
TRADING_DAYS_PER_YEAR = 252


def get_date_to_daily_risk_free_rates():
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
            if date >= TEST_START_DATE and date <= TEST_END_DATE:
                date_to_daily_risk_free_rate[date] = float(row[1])
    return date_to_daily_risk_free_rate


def get_ticker_to_closing_prices():
    """
    Returns a dictionary where keys are stock tickers and
    the items are a list of the closing price of the stock
    from the start date to the end date (in order)

    Also returns a list of dates corresponding to the closing prices
    """
    csvs = [f"{CSVS_REL_PATH}/{f}" for f in os.listdir(CSVS_REL_PATH) if f.endswith(CSV_SUFFIX)]

    # Find CSV row number of test start date
    start_date_row_num = 0
    with open(csvs[0], newline="") as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        rows = rows[1:]  # remove header
        start_date_row_num += 1
        for row in rows:
            if datetime.strptime(row[0], CSV_DATE_FMT) == TEST_START_DATE:
                break
            start_date_row_num += 1

    # Compute closing price of test start and end dates for each stock
    ticker_to_closing_prices = {}
    dates = set()
    for csv_path in csvs:
        ticker = csv_path.lstrip(CSVS_REL_PATH).rstrip(CSV_SUFFIX)
        with open(csv_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader][start_date_row_num:]  # only care about test data
            ticker_to_closing_prices[ticker] = [
                float(row[2]) for row in rows
            ]
            if not dates:
                dates = set([datetime.strptime(row[0], CSV_DATE_FMT) for row in rows])

    dates = sorted(list(dates))
    return dates, ticker_to_closing_prices


def get_buy_and_hold_metrics(ticker_to_closing_prices, daily_rf_rates):
    """
    Given the list of closing prices for each ticker, return the
    annualized return rate and sharpe ratio for those tickers over
    the test period using the buy and hold strategy
    """
    # Compute how many shares we have for each stock
    # NOTE: may want to change this so that remaining balance is used to
    #       buy up cheapest stock? (to minimize amount of unutilized capital)
    ticker_to_share_num = {}
    for ticker, closing_prices in ticker_to_closing_prices.items():
        initial_price = closing_prices[0]
        num_purchased = CAPITAL_PER_STOCK // initial_price
        ticker_to_share_num[ticker] = num_purchased

    # Find daily return rate for overall portfolio
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


def main():
    # Read CSVs
    date_to_daily_risk_free_rate = get_date_to_daily_risk_free_rates()
    dates, ticker_to_closing_prices = get_ticker_to_closing_prices()

    # There were some days when the stock market was open but the US
    # Treasury did not publish yield curve rates
    # For such dates, approximate using the previous day when
    # rates were published
    daily_risk_free_rates = []
    for date in dates:
        if date not in date_to_daily_risk_free_rate:
            missing_date_index = dates.index(date)
            date_to_daily_risk_free_rate[date] = date_to_daily_risk_free_rate[dates[missing_date_index - 1]]
        daily_risk_free_rates.append(date_to_daily_risk_free_rate[date])

    # Compute metrics
    bnh_annualized_return, bnh_sharpe = get_buy_and_hold_metrics(
        ticker_to_closing_prices, daily_risk_free_rates
    )
    print(f"Buy and hold strategy - Annualized Expected return: {bnh_annualized_return}, Sharpe Ratio: {bnh_sharpe}")
    # Buy and hold strategy - Annualized Expected return: 0.24217917278406387, Sharpe Ratio: 1.9550916364541249


if __name__ == "__main__":
    main()
