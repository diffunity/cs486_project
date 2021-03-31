import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime
from datetime import timedelta
import csv
import math
import itertools
import random
import copy

%matplotlib inline
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline

from pprint import pprint

import sys
import os
from typing import List
sys.path.append("../FinRL-Library")

def get_stock_data(start_date:str, end_date:str, stocks_tradable:List[str], tech_indicator_list:List[str]):
    """
    start_date and end_date include the whole period from train, validation to test time periods
    """
    df = YahooDownloader(start_date=start_date,
                         end_date=end_date,
                         ticker_list=stocks_tradable).fetch_data()

    fe = FeatureEngineer(use_technical_indicator=True,
#                         tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                         tech_indicator_list=tech_indicator_list,
                         use_turbulence=False,
                         user_defined_feature=False)

    processed = fe.preprocess_data(df)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])

    processed_full = processed_full.fillna(0)
    return processed_full, list_date

# train = data_split(processed_full, start_training, end_training)
# or just train = get_stock_data(start_date, end_date, stocks_tradable)[0]


def get_date_to_daily_risk_free_rates(start_date:datetime, end_date:datetime, risk_free_rates_csv:str):
    """
    Returns a dictionary of date to daily risk free rate
    from the start date to the end date (in order)
    """
    date_to_daily_risk_free_rate = {}
    with open(
        risk_free_rates_csv, "r"
    ) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            date = datetime.datetime.strptime(row[0], CSV_DATE_FMT)
            if date >= start_date and date <= end_date:
                date_to_daily_risk_free_rate[date] = float(row[1])
    return date_to_daily_risk_free_rate


def fill_missing_daily_rf_rates(date_to_daily_rf_rate, required_dates:List[datetime]):
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

def risk_free_adjusted_sharpe_ratio(daily_returns, daily_rf_rates):
    """
    annual_return and annual_volatility are annualized daily returns
    refer to https://github.com/AI4Finance-LLC/FinRL-Library/blob/master/finrl/env/env_stocktrading.py#L183
    for how "annual returns are obtained"
    """
    assert len(daily_returns) == len(daily_rf_rates), "Trading days do not match!"
    trading_days = len(daily_returns)
    sharpe_ratio = (
        np.sqrt(trading_days) *
        (daily_returns - np.array(daily_rf_rates)).mean() / daily_returns.std()
    )
    
    return sharpe_ratio

# example usage
# risk_free_adjusted_sharpe_ratio(
#     df_account_value["account_value"].pct_change().fillna(0).values,
#     daily_risk_free_rates
# )

def add_sentiments(ndays:int, dataset:dict, df:pd.DataFrame):
    """
    dataset: sentiment dataset
    df: main pandas dataset
    """
    ndays = timedelta(days=ndays)
    sentiment_mean = []
    sentiment_std = []
    print("Begin")
    for e, row in df.iterrows():
        date_formatted = pd.to_datetime(row["date"], format="%Y-%m-%d")
        tic = row["tic"]
        dataset_dates = dataset[tic]["date"].apply(lambda x: x.date())
        if date_formatted.date() in dataset_dates.values:
            score = dataset[tic]["sentiment_score"][(dataset_dates >= (date_formatted.date()-ndays)) && (dataset_dates <=  date_formatted.date())]
            sentiment_mean.append(score.values.mean())
            sentiment_std.append(score.values.std())
        
#         print(f"{tic} mean sentiment {sentiment_mean[-1]}")
#         print(f"{tic} std sentiment {sentiment_std[-1]}")
        else:
            sentiment_mean.append(0)
            sentiment_std.append(0)
    print("Done")
    df["sentiment_mean"] = sentiment_mean
    df["sentiment_std"] = sentiment_std
    return df


