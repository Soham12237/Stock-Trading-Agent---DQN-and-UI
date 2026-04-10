import numpy as np

class TradingEnv:
    def __init__(self, df, initial_balance=10000):
        self.df = df
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.balance = self.initial_balance
        self.shares = 0
        self.net_worth = self.initial_balance
        return self._get_state()

    def _get_state(self):
        row = self.df.iloc[self.step_idx]
        return np.array([
            row['Close'],
            row['SMA'],
            row['EMA'],
            row['RSI'],
            self.balance,
            self.shares
        ], dtype=np.float32)

    def step(self, action):
        price = self.df.iloc[self.step_idx]['Close']

        if action == 1 and self.balance > price:
            self.shares += 1
            self.balance -= price

        elif action == 2 and self.shares > 0:
            self.shares -= 1
            self.balance += price

        prev_worth = self.net_worth
        self.net_worth = self.balance + self.shares * price

        reward = self.net_worth - prev_worth

        self.step_idx += 1
        done = self.step_idx >= len(self.df) - 1

        return self._get_state(), reward, done