import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

import numpy as np
import pandas as pd

from model import PredictionModel
from indicators import add_features, standardize_df, filter_data
from news import merge_df, sentiment_training_df
import warnings

warnings.filterwarnings('ignore')

checkpoints = torch.load("saved_models/2nd_hybrid_eurusd_lstm_model.pth", weights_only=False)
features = checkpoints['features']
scaler = checkpoints['scaler']
model_state = checkpoints['model_state']

model = PredictionModel(len(features))
model.load_state_dict(model_state)
model.eval()

s_df = sentiment_training_df('backtest_fundamentals')

df = pd.read_csv('EURUSD5.csv', sep='\t', header=None)

df = standardize_df(df)
df = add_features(df)
df = filter_data(df).dropna()

df = merge_df(s_df, df)

arima_model = ARIMA(df['close'], order=(2,1,2)).fit()
df['arima_pred'] = arima_model.fittedvalues
df['residuals'] = df['close'] - df['arima_pred']

X = scaler.transform(df[features])

def create_sequence(X, window=60):
    Xs = []
    for i in range(len(X)- window):
        Xs.append(X[i:i+window])
    return np.array(Xs)

X_seq = create_sequence(X)

X_tensor = torch.tensor(X_seq, dtype=torch.float32)

with torch.no_grad():
    opp_pred, dir_pred = model(X_tensor)

    opp_pred = torch.sigmoid(opp_pred).numpy().flatten()
    dir_pred = torch.sigmoid(dir_pred).numpy().flatten()

positions = np.zeros_like(dir_pred)
trade_mask = opp_pred > 0.7

positions[(trade_mask) & (dir_pred > 0.55)] = 1
positions[(trade_mask) & (dir_pred < 0.45)] = -1

close_prices = df['close'].values[-len(positions):]
returns = np.diff(close_prices) / close_prices[:-1]
returns = returns[:len(positions)]

positions = positions[:-1]

cost = 0.0001
turnover = np.abs(np.diff(np.insert(positions, 0, 0)))

strategy_returns = (positions * returns) - (turnover * cost)
equity = np.cumprod(1 + strategy_returns)

sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * np.sqrt(252 * 24 * 4)
accuracy = (np.sign(returns) == positions).mean()

actual_direction = np.sign(returns)

trade_mask = positions != 0

trade_accuracy = (
    positions[trade_mask] == actual_direction[trade_mask]
).mean()

wins = strategy_returns[strategy_returns > 0]
losses = strategy_returns[strategy_returns < 0]

print("Avg Win:", wins.mean())
print("Avg Loss:", losses.mean())
print("Win/Loss Ratio:", abs(wins.mean() / losses.mean()))
print("Trade Accuracy: ", trade_accuracy)
print(f'Model Accuracy: {accuracy:.3f}')
print(f'Sharpe Ratio: {sharpe:.3f}')
print(f'Total Return: {equity[-1]:.3f}')