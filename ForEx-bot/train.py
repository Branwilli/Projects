import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.arima.model import ARIMA

from sklearn.preprocessing import StandardScaler

from model import PredictionModel
from indicators import add_features, add_targets, standardize_df, filter_data, smooth_labels
from news import merge_df, sentiment_training_df
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

s_df = sentiment_training_df('training_fundamentals')

df = pd.read_csv('EURUSD1.csv', sep='\t', header=None)

df = standardize_df(df)
df = add_features(df)
df = add_targets(df)

df = df.dropna()

df = filter_data(df).dropna()

df = merge_df(s_df, df)

arima_model = ARIMA(df['close'], order=(2,1,2)).fit()
df['arima_pred'] = arima_model.fittedvalues
df['residuals'] = df['close'] - df['arima_pred']

features = [
    'volatility','volatility_spike','volume_spike','range_expansion','trend_strength',
    'RSI','momentum_5','htf_bias','ma_10','ma_20',
    'vol_trend','expansion_strength','rsi_mom','trend_bias',
    'residuals','arima_pred', 'sentiment_score'
]

scaler = StandardScaler()
X = scaler.fit_transform(df[features])

y_opp = df['target_opportunity'].values
y_dir = df['target_direction'].values

y_dir = np.nan_to_num(y_dir, nan=0.0)

def create_sequences(X, y1, y2, window_size=60):
  X_s, y1s, y2s = [], [], []
  for i in range(len(X) - window_size):
      X_s.append(X[i:i+window_size])
      y1s.append(y1[i+window_size])
      y2s.append(y2[i+window_size])
  return np.array(X_s), np.array(y1s), np.array(y2s)

X_seq, y1_seq, y2_seq =create_sequences(X, y_opp, y_dir)

split = int(len(X_seq) * 0.8)

X_train = torch.tensor(X_seq[:split], dtype=torch.float32)
y1_train = torch.tensor(y1_seq[:split], dtype=torch.float32)
y2_train = torch.tensor(y2_seq[:split], dtype=torch.float32)

X_test = torch.tensor(X_seq[split:], dtype=torch.float32)

model = PredictionModel(len(features))
pos_weight_opp = (len(y1_train) - y1_train.sum()) / y1_train.sum()

mask = y1_train > 0
pos_weight_dir = (len(y2_train[mask]) - y2_train[mask].sum()) / y2_train[mask].sum()

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_opp)
criterion_dir = nn.BCEWithLogitsLoss(pos_weight=pos_weight_dir)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    patience=3,
    factor=0.5
)

epochs = 120 

for i in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    opp_out, dir_out = model(X_train)
    opp_out = opp_out.squeeze()
    dir_out = dir_out.squeeze()
    
    y1_smooth = smooth_labels(y1_train, 0.08)
    loss_opp = criterion(opp_out, y1_smooth)

    mask = y1_train > 0

    if mask.sum() > 0:
        y2_smooth = smooth_labels(y2_train[mask], 0.15)
        strength = torch.abs(y2_train[mask] - 0.5) * 2
        loss_dir = (criterion_dir(dir_out[mask], y2_smooth) * strength).mean()

    else:
        loss_dir = torch.tensor(0.0)

    target_mean = torch.tensor(0.40)
    calibration_loss = (torch.sigmoid(opp_out).mean() - target_mean) ** 2
    loss = 2.0 * loss_opp + 1.0 * loss_dir + 0.5 * calibration_loss
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    scheduler.step(loss)
    
    print(f"Epoch {i+1}/{epochs} - Loss: {loss.item():.4f} | Opp: {loss_opp.item():.4f} | Dir: {loss_dir.item():.4f}")

model.eval()

with torch.no_grad():
    opp_pred, dir_pred = model(X_test)

    opp_pred = torch.sigmoid(opp_pred).numpy().flatten()
    dir_pred = torch.sigmoid(dir_pred).numpy().flatten()
    
positions = np.zeros_like(dir_pred)
trade_mask = opp_pred > 0.60

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

print('Opportunity Mean: ', opp_pred.mean(), 'Std: ', opp_pred.std())
print('Direction Mean: ', dir_pred.mean(), 'Std: ', dir_pred.std())

os.makedirs("saved_models",exist_ok=True)

torch.save({
    'model_state':model.state_dict(),
    'scaler': scaler,
    'features': features
    },"saved_models/2nd_hybrid_eurusd_lstm_model.pth")

print("\nModel saved to: saved_models/2nd_hybrid_eurusd_lstm_model.pth")