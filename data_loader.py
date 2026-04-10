import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df