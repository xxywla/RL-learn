from typing import List

import gym
import matplotlib.pyplot as plt
from train import Agent


def run() -> List[float]:
    agent = Agent()
    env = gym.make('CartPole-v0')
    reward_history = []

    for epoch in range(3000):
        done = False
        state = env.reset()
        total_reward = 0

        while not done:
            action, prob = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)
            agent.add(action, prob)

            state = next_state
            total_reward += reward

        agent.update()
        reward_history.append(total_reward)
        if (epoch + 1) % 100 == 0:
            avg_reward = sum(reward_history[-100:]) / 100
            print(f'Epoch: {epoch + 1}, Average Reward: {avg_reward}')
    return reward_history


def draw_plot(reward_history: List[float]):
    # 训练结束后绘制奖励变化图
    plt.plot(reward_history)
    plt.xlabel('回合')
    plt.ylabel('总奖励')
    plt.title('每回合奖励')
    plt.grid(True)
    plt.savefig('simple.png')


if __name__ == '__main__':
    rewards = run()
    draw_plot(rewards)
