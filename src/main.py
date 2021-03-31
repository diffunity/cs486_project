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
import time

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
sys.path.append("../FinRL-Library")

from utils import *
from dataloader import *

if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)

configs = yaml.load(open("./conf/config.yml").read(), Loader=yaml.Loader)

def load_and_save():
    stocks_tradable = configs["stocks_tradable"]

    dataset = dict()
    for root, dirs, files in os.walk("./data/sentiments", topdown=False):
        for name in files:
            if name.split("_")[0] in stocks_tradable:
            dataset[name.split("_")[0]] = pd.read_csv(os.path.join(root, name), index_col=0).reset_index(drop=True)
            dataset[name.split("_")[0]]["date"] = pd.to_datetime(dataset[name.split("_")[0]]["date"], format="%Y-%m-%d")

    df = get_stock_data(
    i    configs["train"]["start_date"], configs["test"]["end_date"], configs["stocks_tradable"]

    )

    df = add_sentiments(configs["sentiments"]["days"], dataset, df)

    train = data_split(df, cofings["train"]["start_date"], configs["train"]["end_date"])
    validation = data_split(df, cofings["validation"]["start_date"], configs["validation"]["end_date"])
    train_for_test = data_split(df, configs["train"]["start_date"], configs["validation"]["end_date"])
    test = data_split(df, configs["test"]["start_date"], configs["test"]["end_date"])

    testing_days = pd.Series(test.date.unique())

    train.to_csv("./data/train_data.csv", index=False)
    validation.to_csv("./data/validation_data.csv", index=False)
    train_for_test.to_csv("./data/train_for_test.csv", index=False)
    test.to_csv("./data/test_data.csv", index=False)
    testing_days.to_csv("./data/testing_days.csv", index=False)

    print("train, validation, train_for_test, test files saved")
    
def train():
    train = pd.read_csv("./data/train_data.csv")
    validation = pd.read_csv("./data/validation_data.csv")
    features = [
        ["open", "high", "low", "close", "volume"],
        ["open", "high", "low", "close", "volume"] + ["sentiment_mean", "sentiment_std"],
        config.TECHNICAL_INDICATORS,
        config.TECHNICAL_INDICATORS + ["sentiment_mean", "sentiment_std"]
    ]

    model_names = [
        "OHLCV",
        "OHLCV_sentiments",
        "MACD",
        "MACD_sentiments"
    ]

    batch_sizes = [32, 64, 128]
    learning_rates = [0.0001, 0.001, 0.005, 0.01]

    repetition = 3

    for model_name, feature_set in zip(features, model_names):
        for rep in range(repetition):
            perf_results = dict()
            for batch_size in batch_sizes:
                for lr in learning_rates:
                    
                    ctime = time.time()
                    perf_stats_all, _ = train_configuration(
                        f"{model_name}_{rep}",
                        train,
                        validation,
                        feature_set,
                        batch_sizes,
                        learning_rates,
                        42
                    )


                    perf_results[f"result_{batch_size}_{lr}"] = perf_stats_all.to_json()

            open(f"{model_name}_{rep}.json","w").write(json.dumps(perf_results)) 
            print(f"Results saved to {save_fname}.json") 

            print(f"Time taken {(time.time() - ctime)/60}")
    

def test():
    daily_risk_free_rates = fill_missing_daily_rf_rates(
        date_to_daily_risk_free_rate, [datetime.datetime.strptime(date, "%Y-%m-%d") for date in dates["0"]]
    )
    
    train = pd.read_csv("./data/train_for_test.csv")
    test = pd.read_csv("./data/test_data.csv")


if __name__ == "__main__":
    
