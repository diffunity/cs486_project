import csv
from datetime import datetime
from datetime import timedelta
import pandas as pd
import yfinance as yf

SELECTED_STOCK_TICKERS = [
    "BEN",  # Franklin Resources
    "ROK",  # Rockwell Automation Inc.
    "ALL",  # Allstate Corp.
    "LLY",  # Lilly (Eli) & Co.
    "CTXS",  # Citrix Systems
    "MRO",  # Marathon Oil Corp.
    "SBUX",  # Starbucks Corp.
    "EA",  # Electronic Arts
    "PFE",  # Pfizer Inc.
    "MSFT",  # Microsoft Corp.
]
START_TIME = datetime(2011, 2, 18)
END_TIME = datetime(2021, 2, 19)
DATE_FMT = "%Y-%m-%d"  # Date format for yfinance
CSV_HEADER = ["Date", "Open", "Close", "High", "Low", "Volume", "MA7", "MA30"]


def trading_days_to_regular_days(days):
    # rough estimate--doesn't account for holidays
    return days / 5 * 7


def main():
    for ticker in SELECTED_STOCK_TICKERS:
        # Create the CSV file
        with open(f"csvs/{ticker}_technical_data.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

            # Go back 30 more (trading) days to calculate moving average
            data = yf.download(
                ticker,
                start=(
                    START_TIME - timedelta(days=trading_days_to_regular_days(30))
                ).strftime(DATE_FMT),
                end=END_TIME.strftime(DATE_FMT),
                interval="1d",
                auto_adjust=True,
            )

            # Maintain list of closing prices for MA calculation
            last_30_days_closing_prices = []
            for i, closing_price in enumerate(data["Close"].head(30)):
                last_30_days_closing_prices.append(closing_price)

            # Remove first 30 days data
            data = data.iloc[30:]
            for ts, row in data.iterrows():
                date = datetime(year=ts.year, month=ts.month, day=ts.day)

                # Calculate moving averages and update closing price list
                ma7 = sum(last_30_days_closing_prices[-7:]) / 7
                ma30 = sum(last_30_days_closing_prices) / 30
                last_30_days_closing_prices.pop(0)
                last_30_days_closing_prices.append(row["Close"])

                # Write data to CSV
                csv_row = [date] + list(row) + [ma7, ma30]
                writer.writerow(csv_row)


if __name__ == "__main__":
    main()
