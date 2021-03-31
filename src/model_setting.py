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
import json
import sys
import os
from pprint import pprint

%matplotlib inline
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline

sys.path.append("../FinRL-Library")

def train_configuration(
    model_name: str
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: List[str],
    batch_size: int,
    lr: float,
    seed: int,
    ):

    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(features)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 1000000, 
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": features,
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4,
        "model_name": model_name 
    }

    e_train_gym = StockTradingEnv(df = train, **env_kwargs)
    e_train_gym.seed(42)
    e_train_gym.action_space.seed(42)

    env_train, _ = e_train_gym.get_sb_env()
    env_train.seed(seed)
    env_train.action_space.seed(seed)
    print(type(env_train))

    agent = DRLAgent(env = env_train)
    model_ddpg = agent.get_model("ddpg", 
                                 model_kwargs={"batch_size": batch_size, 
                                                "buffer_size": 50000, 
                                                "learning_rate": lr}
                                  )
    trained_ddpg = agent.train_model(model=model_ddpg,
                                     tb_log_name='ddpg',
                                     total_timesteps=50000)

    e_trade_gym = StockTradingEnv(df = validation, **env_kwargs)
    e_trade_gym.seed(seed)
    e_trade_gym.action_space.seed(seed)
    df_account_value, df_actions = DRLAgent.DRL_prediction(
          model=trained_ddpg,
          environment = e_trade_gym
    )
        
    print("==============Get Backtest Results===========")
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

    perf_stats_all = backtest_stats(account_value=df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')
       
    return perf_stats_all, df_account_value

