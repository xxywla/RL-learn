import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from model import Policy, ValueNet


class Agent():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        # 收益的折扣因子
        self.gamma = 0.99

        self.lr_policy = 0.0002  # 学习率
        self.lr_value = 0.0002

        self.action_size = 2  # 动作空间大小，共两个动作：向左推和向右推

        # 策略
        self.pi = Policy(self.action_size)
        # 价值网络
        self.value = ValueNet()

        # 使用 Adam 优化器
        self.optimizer_policy = optim.Adam(self.pi.parameters(), lr=self.lr_policy)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=self.lr_value)

    def get_action(self, state):
        # 将状态转换成torch.tensor类型
        state = torch.tensor(state[np.newaxis, :])
        prob = self.pi(state)[0]
        m = Categorical(prob)
        action = m.sample().item()
        return action, prob[action]

    def update(self, state, prob, next_state, reward, done):
        # 由蒙特卡罗采样变成时序查分采样
        state = torch.tensor(state[np.newaxis, :])
        next_state = torch.tensor(next_state[np.newaxis, :])
        # 未来价值 贝尔曼方程
        target = reward + self.gamma * self.value(next_state) * (1 - done)
        target.detach()
        # 当前状态价值
        v = self.value(state)

        # 价值函数的损失
        loss_fn = nn.MSELoss()
        loss_value = loss_fn(v, target)

        # 策略函数的损失
        delta = target - v
        loss_policy = -torch.log(prob) * delta.item()

        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()

        loss_value.backward()
        loss_policy.backward()

        self.optimizer_value.step()
        self.optimizer_policy.step()
