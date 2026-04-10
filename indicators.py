import pandas as pd

def add_indicators(df):
    df['SMA'] = df['Close'].rolling(window=10).mean()
    df['EMA'] = df['Close'].ewm(span=10).mean()

    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    df.fillna(0, inplace=True)
    return df