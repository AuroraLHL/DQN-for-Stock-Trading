'''
Author: Hongliang Lu, lhl@pku.edu.cn
Date: 2024-06-27 13:50:11
LastEditTime: 2024-06-27 14:15:10
FilePath: /DQN for Stock Trading/model.py
Description: The model of the DQN network.
Organization: College of Engineering,Peking University.
'''



import torch
import torch.nn as nn
import torch.nn.functional as F
    
class QNetwork(nn.Module):
    """QNetwork (Deep Q-Network), state is the input, 
        and the output is the Q value of each action.
    """
    def __init__(self, state_size, action_size, fc1_units=128, fc2_units=128, fc3_units=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size , fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
        self.dropout = nn.Dropout(0.1)  # Dropout with 20% probability

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x