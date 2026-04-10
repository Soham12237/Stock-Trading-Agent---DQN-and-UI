# Stock-Trading-Agent---DQN-and-UI
Deep Q-Network agent that autonomously trades AAPL stock (BUY/HOLD/SELL) using RL — served via Flask REST API with a live web UI.

# 📈 RL-Powered Stock Trading Agent

> A Deep Q-Network (DQN) agent that learns to autonomously **BUY**, **HOLD**, or **SELL** Apple (AAPL) stock by interacting with a custom trading environment — and exposes its decisions through a live web UI.

---

## 🧠 Overview

This project applies **Deep Reinforcement Learning** to algorithmic trading. The agent observes a 6-dimensional market state (price, technical indicators, portfolio status) and learns a policy that maximises net portfolio worth over time using the DQN algorithm with experience replay and ε-greedy exploration.

After training, the model is **persisted to disk** and served via a **Flask REST API** with a single-page HTML/CSS/JS dashboard for real-time inference.

---

## 🗂️ Project Structure

```
rl-trading-agent/
│
├── AAPL.csv              # Historical OHLCV data for Apple Inc.
│
├── data_loader.py        # Loads and cleans CSV data
├── indicators.py         # Computes SMA, EMA, RSI from price data
├── environment.py        # Custom OpenAI-Gym-style trading environment
├── dqn_agent.py          # DQN model, agent logic, save/load/predict
├── train.py              # Training loop — saves checkpoint on completion
│
├── A2C.py                # A2C + REINFORCE on CartPole-v1 (comparison)
├── utils.py              # Plotting utility
│
├── app.py                # Flask server — serves /predict and /model-info
├── index.html            # Single-page web UI for live inference
│
├── dqn_model.pth         # Saved model checkpoint (generated after training)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/rl-trading-agent.git
cd rl-trading-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the agent
```bash
python train.py
```
This will train the DQN agent on `AAPL.csv` and save a full checkpoint to `dqn_model.pth`.

> **Tip:** Increase `episodes` and the `df.head(300)` row limit in `train.py` for better results.

### 4. Launch the web UI
```bash
python app.py
```
Then open **http://localhost:5000** in your browser.

---

## 🖥️ Web Interface

The dashboard accepts live market state inputs and returns the agent's decision in real time:

| Input Field | Description |
|-------------|-------------|
| Close Price | Latest closing price of the stock |
| SMA | 10-period Simple Moving Average |
| EMA | 10-period Exponential Moving Average |
| RSI | 14-period Relative Strength Index (0–100) |
| Balance | Current cash balance in portfolio |
| Shares Held | Number of shares currently held |

**Output:** `BUY` / `HOLD` / `SELL` decision with a Q-value confidence bar chart.

---

## 🤖 Algorithm Details

| Component | Detail |
|-----------|--------|
| Algorithm | Deep Q-Network (DQN) |
| State size | 6 (Close, SMA, EMA, RSI, Balance, Shares) |
| Action space | 3 — HOLD (0), BUY (1), SELL (2) |
| Reward | Change in net portfolio worth per step |
| Network | FC(6→64) → ReLU → FC(64→64) → ReLU → FC(64→3) |
| Optimizer | Adam (lr = 0.001) |
| Exploration | ε-greedy decay (ε₀ = 1.0 → ε_min = 0.01, decay = 0.995) |
| Replay buffer | Deque, capacity 10,000 |
| Batch size | 32 |

---

## 📊 Dataset

- **Stock:** Apple Inc. (AAPL)
- **Columns used:** Date, Open, High, Low, Close, Volume
- **Derived features:** SMA(10), EMA(10), RSI(14)
- **Source:** [Yahoo Finance](https://finance.yahoo.com) / your own OHLCV CSV

---

## 🔌 API Reference

### `POST /predict`
```json
// Request
{
  "close": 182.50,
  "sma": 179.30,
  "ema": 180.10,
  "rsi": 58.4,
  "balance": 10000,
  "shares": 2
}

// Response
{
  "action": 1,
  "label": "BUY",
  "q_values": [-0.0312, 0.2847, -0.1053]
}
```

### `GET /model-info`
Returns loaded model metadata: epsilon, state size, action size.

---

## 🚀 Future Work

- [ ] Extend to multi-stock portfolio management
- [ ] Implement PPO / A3C for comparison
- [ ] Integrate live market data (Yahoo Finance / Alpha Vantage API)
- [ ] Add training progress dashboard (loss curves, net worth over episodes)
- [ ] Dockerise the Flask app for deployment

---

## 👥 Team

| Name | ID |
|------|----|

| Soham Debnath | 23BAI0084 |
| Tejas K M | 23BAI0105 |
| Nischay Nalamanda | 23BAI0111 |

**Department:** Department of computer science  
**College:** VIT Vellore

---

## 📄 License

This project is for academic purposes. 
