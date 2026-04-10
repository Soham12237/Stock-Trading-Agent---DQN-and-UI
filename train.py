from data_loader import load_data
from indicators import add_indicators
from environment import TradingEnv
from dqn_agent import DQNAgent

df = load_data("AAPL.csv")
df = add_indicators(df)
df = df.head(300)  
env = TradingEnv(df)

state_size = 6
action_size = 3

agent = DQNAgent(state_size, action_size)

episodes = 5

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    step_count = 0

    while True:
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        total_reward += reward
        step_count += 1

        if step_count % 50 == 0:
            print(f"Step {step_count} | Net Worth: {env.net_worth:.2f}")

        if done:
            print(f"Episode {ep+1} finished | Total Reward: {total_reward:.2f} | Net Worth: {env.net_worth:.2f}")
            break


agent.save("dqn_model1.pth")