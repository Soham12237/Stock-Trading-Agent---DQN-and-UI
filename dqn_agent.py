import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        for s, a, r, ns, done in batch:
            target = r
            if not done:
                target += self.gamma * torch.max(
                    self.target_model(torch.FloatTensor(ns))
                ).item()

            target_f = self.model(torch.FloatTensor(s))
            target_f[a] = target

            loss = nn.MSELoss()(self.model(torch.FloatTensor(s)), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path="dqn_model.pth"):
        """Save model weights and agent state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_size': self.state_size,
            'action_size': self.action_size,
        }, path)
        print(f"[✓] Model saved to {path}")

    @classmethod
    def load(cls, path="dqn_model.pth"):
        """Load a saved agent from disk."""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        agent = cls(checkpoint['state_size'], checkpoint['action_size'])
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        agent.model.eval()
        print(f"[✓] Model loaded from {path}")
        return agent

    def predict(self, state):
        """
        Deterministic prediction (no exploration).
        Returns: action int and label string.
        """

        # unpack state (adjust indices if your order is different)
        close, sma, ema, rsi, balance, shares_held = state

        # 🔥 HARD RULE: force BUY
        if balance > 0 and shares_held == 0:
            return 1, "BUY", [0.0, 1.0, 0.0]

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze().tolist()

        action = int(torch.argmax(torch.FloatTensor(q_values)).item())
        label = {0: "HOLD", 1: "BUY", 2: "SELL"}[action]
        return action, label, q_values