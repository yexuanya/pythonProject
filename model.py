import ccxt
import time
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

# 初始化 Binance 交易所
exchange = ccxt.binance({
    'enableRateLimit': True,  # 启用速率限制
    'proxies': {
        'http': 'http://127.0.0.1:15236',  # 设置 HTTP 代理
        'https': 'http://127.0.0.1:15236'  # 设置 HTTPS 代理
    }
})

# 定义强化学习环境
class PriceEnv(gym.Env):
    def __init__(self):
        super(PriceEnv, self).__init__()

        self.action_space = gym.spaces.Discrete(3)  # 0: 卖出, 1: 持有, 2: 买入
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)  # 假设8个特征

        self.balance = 1000  # 初始资金
        self.stock_owned = 0  # 当前持有的股票数量
        self.current_step = 0
        self.data = []  # 用于存储实时获取的数据

    def reset(self):
        self.balance = 1000
        self.stock_owned = 0
        self.current_step = 0
        return self.get_state()

    def get_state(self):
        # 返回当前的市场状态
        return np.array(self.data[-1] if len(self.data) > 0 else [0]*8)

    def step(self, action):
        if len(self.data) == 0:
            return self.get_state(), 0, False, {}

        current_data = self.data[-1]
        current_price = current_data[6]  # 假设 'last' 是数据的最后一项

        # 动作逻辑：买入、卖出、持有
        reward = 0
        if action == 0:  # 卖出
            reward = self.stock_owned * (current_price - self.data[-2][6] if len(self.data) > 1 else 0)
            self.balance += reward
            self.stock_owned = 0
        elif action == 2:  # 买入
            if self.balance >= current_price:
                self.stock_owned += 1
                self.balance -= current_price
            else:
                reward = -1  # 资金不足，惩罚

        done = False
        if self.current_step == len(self.data) - 1:
            done = True

        self.current_step += 1
        return self.get_state(), reward, done, {}

    def update_data(self, new_data):
        self.data.append(new_data)


# 定义 Q 网络
class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.dense1 = Dense(64, activation='relu', input_shape=(state_size,))
        self.dense2 = Dense(64, activation='relu')
        self.dense3 = Dense(action_size, activation='linear')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.target_model.set_weights(self.model.get_weights())
        self.memory = deque(maxlen=2000)
        self.optimizer = Adam(learning_rate=0.001)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.update_target_freq = 10
        self.steps = 0

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(state)  # 计算 Q 值
        return np.argmax(act_values[0])  # 返回最大 Q 值对应的动作

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)

        with tf.GradientTape() as tape:
            q_values = self.model(states)  # (batch_size, action_size)
            next_q_values = self.target_model(next_states)  # (batch_size, action_size)

            next_q_values_max = np.max(next_q_values, axis=1)  # (batch_size,)

            # 确保 rewards 是一维的，并与 target_q_values 对应
            # target_q_values = rewards + self.gamma * next_q_values_max * (1 - np.array(dones))  # (batch_size,)
            target_q_values = rewards + self.gamma * next_q_values_max * (1 - np.array(dones).astype(np.float32))

            # 通过 actions 获取对应的 Q 值
            q_value = tf.gather(q_values, actions, axis=1, batch_dims=1)  # 获取每个动作的 Q 值

            loss = tf.reduce_mean(tf.square(target_q_values - q_value))  # 均方误差损失

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # epsilon-greedy 策略
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.steps % self.update_target_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        self.steps += 1

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


# 实时获取市场数据并更新环境
def fetch_realtime_data(symbol, interval=1):
    env = PriceEnv()
    agent = DQNAgent(state_size=8, action_size=3)  # 假设有8个特征

    try:
        while True:
            # 获取实时数据
            ticker = exchange.fetch_ticker(symbol)
            print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"最新价格: {ticker['last']} USDT")

            # 模拟数据，假设我们用以下数据：['high', 'low', 'bid', 'bidVolume', 'ask', 'askVolume', 'last', 'percentage']
            market_data = [ticker['high'], ticker['low'], ticker['bid'], ticker['bidVolume'], ticker['ask'],
                           ticker['askVolume'], ticker['last'], ticker['percentage']]
            env.update_data(market_data)  # 更新环境数据

            state = env.reset()  # 获取初始状态
            state = np.reshape(state, [1, 8])  # 假设状态空间是8维

            done = False
            total_reward = 0

            # 执行训练步骤
            while not done:
                action = agent.act(state)  # 基于当前状态选择动作
                next_state, reward, done, _ = env.step(action)  # 执行动作
                next_state = np.reshape(next_state, [1, 8])  # 处理下一个状态

                agent.remember(state, action, reward, next_state, done)
                agent.replay()  # 回放经验并训练模型

                state = next_state  # 更新状态
                total_reward += reward

            print(f"Total Reward: {total_reward}")
            time.sleep(interval)  # 每秒钟获取一次数据

    except KeyboardInterrupt:
        print("手动停止实时数据获取")

# 启动实时数据获取并训练
fetch_realtime_data('BTC/USDT', interval=1)
