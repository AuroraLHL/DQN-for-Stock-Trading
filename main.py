'''
Author: Hongliang Lu, lhl@pku.edu.cn
Date: 2024-06-27 13:43:03
LastEditTime: 2024-06-27 13:49:17
FilePath: /stockPrediction-master/main.py
Description: 
Organization: College of Engineering,Peking University.
'''
from dqn_agent import Agent
from model import QNetwork
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from StockExchange import StockExchange

STATE_SIZE = 10 # 状态空间大小
EPISODE_COUNT = 100 # episode 数量

data_dir = "StockData"
filename = "GOOGL.csv"
file_dir = data_dir + "/" + filename

# 使用 pandas 读取 CSV 文件
try:
    df = pd.read_csv(file_dir, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_dir, encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv(file_dir, encoding='iso-8859-1')

# 提取收盘价
stockData = list(df['Close'].values)

# 初始化 agent
STATE_SIZE = 10
agent = Agent(state_size=STATE_SIZE, action_size=3)
l = len(stockData) - 1

stock_agent = StockExchange(stockData, agent, STATE_SIZE)

stock_agent.train(episodes=EPISODE_COUNT,filename=filename)

stock_agent.test()

stock_agent.plot_result()