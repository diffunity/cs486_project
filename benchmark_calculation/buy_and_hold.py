import os
import csv
from datetime import datetime
import numpy as np

TEST_START_DATE = datetime(2019, 12, 19)
TEST_END_DATE = datetime(2021, 2, 18)
CSVS_REL_PATH = "../data_preprocessing/csvs"
CSV_SUFFIX = "_technical_data.csv"
CSV_DATE_FMT = "%Y-%m-%d %H:%M:%S"
CAPITAL_PER_STOCK = 10000  # USD


def get_ticker_to_closing_price_list():
    """
    Returns a dictionary where keys are stock tickers and
    the items are lists of the closing price of the stock
    from the start date to the end date (in order)
    """

    csvs = [f"{CSVS_REL_PATH}/{f}" for f in os.listdir(CSVS_REL_PATH)]

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
    ticker_to_closing_price_list = {}
    for csv_path in csvs:
        ticker = csv_path.lstrip(CSVS_REL_PATH).rstrip(CSV_SUFFIX)
        with open(csv_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader][start_date_row_num:]  # only care about test data
            ticker_to_closing_price_list[ticker] = [
                float(row[2]) for row in rows
            ]

    return ticker_to_closing_price_list


def get_buy_and_hold_metrics(ticker_to_closing_price_list):
    """
    Given the list of closing prices, return the expected rate of return
    and standard deviation over the test period using the buy and hold
    strategy.
    """
    # Compute number of shares held for each stock
    # NOTE: may want to change this so that remaining balance is used to
    #       buy up cheapest stock? (so that there is no unutilized capital)
    ticker_to_share_num = {}
    for ticker, closing_price_list in ticker_to_closing_price_list.items():
        initial_price = closing_price_list[0]
        num_purchased = CAPITAL_PER_STOCK // initial_price
        ticker_to_share_num[ticker] = num_purchased

    # Get buy and hold metrics
    daily_return_rates = []
    num_trading_days = len(list(ticker_to_closing_price_list.values())[0])
    for i in range(num_trading_days - 1):
        # Compute daily profit
        daily_total_profit = 0.0
        for ticker, closing_price_list in ticker_to_closing_price_list.items():
            price_today = closing_price_list[i]
            price_tomorrow = closing_price_list[i + 1]
            profit = (price_tomorrow - price_today) * ticker_to_share_num[ticker]
            daily_total_profit += profit

        # Divide by initial capital to get rate of return (NOTE: not sure if this the correct formula..)
        daily_return_rate = daily_total_profit / (CAPITAL_PER_STOCK * len(ticker_to_closing_price_list))
        daily_return_rates.append(daily_return_rate)

    return np.mean(daily_return_rates), np.std(daily_return_rates)


def main():
    ticker_to_closing_price_list = get_ticker_to_closing_price_list()

    print(get_buy_and_hold_metrics(ticker_to_closing_price_list))  # (0.0006957621524140783, 0.01335452332304571)


if __name__ == "__main__":
    main()
