'''
Author: Hongliang Lu, lhl@pku.edu.cn
Date: 2024-05-31 16:10:12
LastEditTime: 2024-06-26 12:03:36
FilePath: /stockPrediction-master/StockExchange.py
Description: 
@Organization: College of Engineering,Peking University.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class StockExchange:
    
    def __init__(self, stockData, agent, state_size):
        self.train_data = stockData[:int(0.9 * len(stockData))]
        self.test_data = stockData[int(0.9 * len(stockData)):]
        self.agent = agent
        self.state_size = state_size
        self.stock_length = len(self.train_data)

    def train(self,episodes, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        scores = []
        
        for i_episode in range(1, episodes+1):
            print("Episode" + str(i_episode))
            state = self.getState(self.train_data, 0, self.state_size + 1)
            total_profit = 0
            self.agent.balance = []
            eps = eps_start

            for t in range(self.stock_length):
                action = self.agent.act(state, eps)
                next_state = self.getState(self.train_data, t + 1, self.state_size + 1)
                reward = 0
                
                if action == 1:# 买入
                    self.agent.balance.append(self.train_data[t])
                elif action == 2 and len(self.agent.balance) > 0: # 卖出
                    bought_price = self.agent.balance.pop(0)
                    total_profit += self.train_data[t] - bought_price
                    reward = self.train_data[t] - bought_price
        
                done = 1 if t == self.stock_length - 1 else 0
                self.agent.step(state, action, reward, next_state, done)
                eps = max(eps_end, eps * eps_decay)
                state = next_state
                if done:
                    print("------------------------------")
                    print("total_profit = " + str(total_profit))
                    print("------------------------------")
                    
            scores.append(total_profit)
        return scores

    def getState(self, data, t, n):
        d = t - n + 1
        block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]
        
        # 确保 block 的长度为 n
        if len(block) < n:
            block = [data[0]] * (n - len(block)) + block
        
        res = []
        for i in range(n - 1):
            res.append(block[i + 1] - block[i])
        return np.array([res])

    
    def test(self):
        l = len(self.test_data)-1
        window_size = 10
        state = self.getState(self.test_data, 0, window_size + 1)
        total_profit = 0
        self.agent.balance = []
        self.action_list = []
        value_list = []
        for t in range(l):
            action = self.agent.act(state, eps=0)
            next_state = self.getState(self.test_data, t + 1, self.state_size + 1)
            if action == 1:# 买入
                self.agent.balance.append(self.test_data[t])
            elif action == 2 and len(self.agent.balance) > 0: # 卖出
                bought_price = self.agent.balance.pop(0)
                total_profit += self.test_data[t] - bought_price
            done = 1 if t == l - 1 else 0
            state = next_state
            self.action_list.append(action)
            value_list.append(self.test_data[t])
            if done:
                print("------------------------------")
                print("total_profit = " + str(total_profit))
                print("------------------------------")
                #plt.plot(np.arange(len(value_list)), value_list)
                self.action_list.append(0)
                
    def plot_result(self):
        # self.action_list.append(0)
        df = pd.DataFrame(self.test_data, columns=['收盘'])
        df['action'] = pd.DataFrame(self.action_list).values

        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(df.index.values, df["收盘"])
        sell = (df['action'].values == 2)
        plt.scatter(df.index[sell], df["收盘"].values[sell], c='r')
        buy = (df['action'].values == 1)
        plt.scatter(df.index[buy], df["收盘"].values[buy], c='g')
        plt.legend(['value', 'sell', 'buy'])
        plt.show()
    
    

            
