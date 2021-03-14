import csv
import math
from datetime import datetime

READ_DATE_FMT = "%m/%d/%Y"
WRITE_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def cmt_rate_to_apy(cmt_rate):
    """
    Convert the CMT (semiannual bond equivalent) yield published on
    the US Treasury website to annual percentage yield.

    See "ARE THE CMT YIELDS ANNUAL YIELDS?" section in
    https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/interest-rates-frequently-asked-questions
    """
    return (1 + cmt_rate / 2) ** 2 - 1.0


def main():
    # US Treasury daily yield curve rates obtained from
    # https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/textview.aspx?data=yield

    # We will use the one month cmt rate as a benchmark for short term risk free rate
    # This is commonly done in adjustable-rate morgages.
    # (https://www.investopedia.com/terms/c/cmtindex.asp)
    with open("csvs/US_treasury_daily_yield_rates.csv", "r") as rf, \
            open("csvs/US_treasury_daily_risk_free_rates.csv", "w") as wf:
        reader = csv.reader(rf)
        writer = csv.writer(wf)

        # Deal with headers
        next(reader)
        writer.writerow(["Date", "Daily_RF_Rate"])

        for row in reader:
            date_str = row[0]
            if row[1] != "N/A":
                # if 1 month cmt rate not published for the day,
                # just assume the same rate as previous day
                one_mo_cmt_rate = float(row[1])
                one_mo_cmt_rate /= 100

            # reformat date
            date_str = datetime.strptime(date_str, READ_DATE_FMT).strftime(WRITE_DATE_FMT)

            # convert 1 month cmt rate to daily rate
            apy = cmt_rate_to_apy(one_mo_cmt_rate)
            daily_rate = (1 + apy) ** (1 / 365)  # assume 365 days in year
            daily_rate -= 1

            writer.writerow([date_str, daily_rate])


if __name__ == "__main__":
    main()
