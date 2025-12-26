import numpy as np
import gym
import torch
import torch.optim as optim
from torch.distributions import Categorical
from model import Policy


class Agent():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        # 收益的折扣因子
        self.gamma = 0.99
        self.lr = 0.0002  # 学习率
        self.action_size = 2  # 动作空间大小，共两个动作：向左推和向右推
        # 策略
        self.pi = Policy(self.action_size)
        # 使用 Adam 优化器
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)
        # 存放每一步的奖励
        self.memory = []

    def get_action(self, state):
        # 将状态转换成torch.tensor类型
        state = torch.tensor(state[np.newaxis, :])
        prob = self.pi(state)[0]
        m = Categorical(prob)
        action = m.sample().item()
        return action, prob[action]

    def add(self, reward, prob):
        self.memory.append((reward, prob))

    def update(self):
        G, loss = 0, 0
        for reward, _ in reversed(self.memory):
            G = reward + self.gamma * G
        for reward, prob in self.memory:
            loss += -torch.log(prob) * G
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []
