import os
import sys
from dotenv import load_dotenv
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA
import time
from datetime import datetime
import traceback

from trade import conn, load_data, get_latest, has_position, place_trade, trade_pullback, manage_pullback_exit, close_position
from logs import log
from indicators import standardize_df, compute_rsi, add_features, filter_data, update_features, append_new_candle
from model import PredictionModel
import warnings

warnings.filterwarnings('ignore')

load_dotenv()

SEQUENCE_LENGTH = os.getenv('SEQUENCE_LENGTH')
OPP_THRESHOLD = os.getenv('OPP_THRESHOLD')
BUY_THRESHOLD = os.getenv('BUY_THRESHOLD')
SELL_THRESHOLD = os.getenv('SELL_THRESHOLD')
SYMBOL =os.getenv('SYMBOL')


class PredictionModel(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.opportunity_head = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.direction_head = nn.Linear(128, 1)
        #self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(out)
        out = self.dropout(out)
        opp = self.opportunity_head(out)
        direction = self.direction_head(out)
        return opp, direction
    
checkpoints = torch.load("saved_models/2nd_hybrid_eurusd_lstm_model.pth", weights_only=False)
features = checkpoints['features']
scaler = checkpoints['scaler']
model_state = checkpoints['model_state']

model = PredictionModel(len(features))
model.load_state_dict(model_state)
model.eval()

def predict(df):
    arima_model = ARIMA(df['close'], order=(2,1,2)).fit()
    df['arima_pred'] = arima_model.fittedvalues
    df['residuals'] = df['close'] - df['arima_pred']

    X = df[features].values[-60:]
    X = scaler.transform(X)

    X_tensor = torch.tensor(X).unsqueeze(0).float()

    with torch.no_grad():
        opp, direction = model(X_tensor)
        opp = torch.sigmoid(opp).item()

        direction = torch.sigmoid(direction).item()

    return opp, direction

if __name__ == '__main__':
    
    conn()

    account_info = mt5.account_info()
    balance = account_info.balance 
    symbol_info = mt5.symbol_info(SYMBOL)
    point = symbol_info.point
    min_stop = symbol_info.trade_stops_level * point

    df = standardize_df(load_data())
    df = add_features(df)
    df = filter_data(df)

    log(f'Opening Balance: {balance}')
    log('\n----------------------BOT STARTED-------------------------\n')
    last_candle_time = None
    try:
        while True:
            try: 
                new = get_latest()
                new = standardize_df(new)
                
                if new.index[-1] not in df.index:
                    df = append_new_candle(df, new)
                    df = df.tail(2000)

                    latest = df.iloc[-1]
                    if latest.isna().any():
                        log('Skipping due to NaN in latest features')
                        continue

                    df = df.dropna()
                    print('Sending data into model\n')
                        
                    opp, direction = predict(df)
                    bias = df['htf_bias'].iloc[-1]

                    log(f'Opp: {opp:.3f} | Dir: {direction:.3f} | Bias: {bias}')
                    position = has_position()
                     
                    if not position:
                        print('No Orders....\n')
                        if opp > float(OPP_THRESHOLD):
                            if direction > float(BUY_THRESHOLD) and bias == 1:
                                place_trade('BUY', balance, min_stop)
                                log('BUY order placed\n')

                            elif direction < float(SELL_THRESHOLD) and bias == 0:
                                place_trade("SELL", balance, min_stop)
                                log('SELL order placed\n')

                            elif( direction > float(BUY_THRESHOLD) and bias == 0) or (direction < float(SELL_THRESHOLD) and bias == 1):
                                trade_pullback(df, opp, direction, min_stop, balance, bias)
                                log('PULLBACK order placed\n')
                    
                    else:
                        print("Orders found\n")
                        manage_pullback_exit(position, df)
                        account_info = mt5.account_info()
                        print(f'\nAvaliable Balance: {account_info.balance}\n')

                    print('Waiting on new 15 minute candle....')
                    time.sleep(900)
                    
            except Exception as e:
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                log(f"""
                        #ERROR: {e},
                        #File Name: {tb.filename}
                        #Line: {tb.lineno},
                        #Function: {tb.name}
                        #Code: {tb.line}
                    """)
                time.sleep(900)
    
    except KeyboardInterrupt:
        log('Manual stop triggered (Ctrl+C)')

    finally:
        log('Shutting down bot...')
        mt5.shutdown()