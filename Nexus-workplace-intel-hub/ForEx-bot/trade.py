import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from logs import log
import MetaTrader5 as mt5

load_dotenv()

LOGIN = os.getenv('LOGIN')
PASSWORD = os.getenv('PASSWORD')
SERVER = os.getenv('SERVER')
SYMBOL =os.getenv('SYMBOL')
SEQUENCE_LENGTH = os.getenv('SEQUENCE_LENGTH')

RISK_PER_TRADE = os.getenv('RISK_PER_TRADE')
SL_PIPS = os.getenv('SL_PIPS')
TP_PIPS = os.getenv('TP_PIPS')

def conn():
    if not mt5.initialize():
        raise RuntimeError('MT5 init failed')

    login = LOGIN
    password = PASSWORD
    server = SERVER

    mt5.login(login, password, server)

    log('Connected!\n')

def load_data(symbol='EURUSD', n=30000):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n)

    if rates is None:
        print('MT5 Error: ', mt5.last_error())
        return pd.DataFrame()

    df = pd.DataFrame(rates)

    if df.empty:
        print('No data returned')
        return df
    
    return df

def get_latest(symbol='EURUSD'):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 1, 3)

    if rates is None:
        print('MT5 Error: ', mt5.last_error())
        return pd.DataFrame()

    df = pd.DataFrame(rates)

    if df.empty:
        print('No data returned')
        return df
    
    return df

def calculate_lot(balance, risk_pct, sl_pip):
    risk_amount = balance * risk_pct
    pip_value = 10
    lot = risk_amount / (sl_pip * pip_value)
    return round(max(lot, 0.01), 2)
 
def has_position():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions:
        return positions[0]
    return None

def close_position(position):
    pnl = calculate_pnl()

    order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY

    symbol_info = mt5.symbol_info_tick(position.symbol)
    price = symbol_info.bid if order_type == mt5.ORDER_TYPE_SELL else symbol_info.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,  
        "price": price,
        "deviation": 20,
        "magic": 123456,
        "comment": "Close via Script",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to close position {position.ticket}: {result.comment}")
    
    log(f"Position {position.ticket} closed successfully | Lot: {position.volume}")
    return pnl

def place_trade(direction, balance, min_stop):
    lot = calculate_lot(float(balance), float(RISK_PER_TRADE), float(SL_PIPS))

    tick = mt5.symbol_info_tick(SYMBOL)
    sl_dist = max(min_stop, float(SL_PIPS) * 0.0001)
    tp_dist = max(min_stop, float(TP_PIPS) * 0.0001)

    if direction == 'BUY':
        price = float(tick.ask)
        sl = price - sl_dist
        tp = price + tp_dist
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = float(tick.bid)
        sl = price + sl_dist
        tp = price - tp_dist
        order_type = mt5.ORDER_TYPE_SELL

    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': SYMBOL,
        'volume': lot,
        'type': order_type,
        'price': price,
        'sl': sl,
        'tp': tp,
        'deviation': 20,
        'magic': 123456,
        'comment': 'AI Trade',
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_FOK
    }
    print('Attempting to place trade....')
    result = mt5.order_send(request)

    if result is None:
        log(f'Order failed: {mt5.last_error()}')

    log(f"TRADE {direction} | Lot: {lot} | Result: {result.retcode}")

def trade_pullback(df, opp, direction, min_stop, balance, bias, 
                   opp_threshold=0.65, strong_dir=0.6, weak_dir=0.5):
    
    last_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]

    if opp <= opp_threshold:
        return 
    
    sell_cond = (
        (bias == 0 and direction > strong_dir and last_close < prev_close) or
        (bias == 1 and direction > strong_dir and last_close < prev_close) or
        (bias == 1 and direction < weak_dir and last_close < prev_close)
    )

    buy_cond = (
        (bias == 0 and direction > strong_dir and last_close > prev_close) or
        (bias == 0 and direction < weak_dir and last_close > prev_close) or
        (bias == 1 and direction < weak_dir and last_close > prev_close)
    )

    log(f"SellCond: {sell_cond} | BuyCond: {buy_cond}")

    if sell_cond:
        log(f"PULLBACK SELL | Opp: {opp:.3f} | Dir: {direction:.3f}\n")
        place_trade('SELL', balance, min_stop)

    elif buy_cond:
        log(f"PULLBACK BUY | Opp: {opp:.3f} | Dir: {direction:.3f}\n")
        place_trade('BUY', balance, min_stop)

def manage_pullback_exit(position, df):
    direction = position.type

    last_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    past_close = df['close'].iloc[-3]

    if direction == 0:
        print('BUY order identified')
        if ((last_close > prev_close) or (prev_close > past_close)):
            print('Preparing to close order....\n')
            close_position(position)

    elif direction == 1:
        print('SELL order identified')
        if ((last_close < prev_close) or (prev_close < past_close)):
            print('Preparing to close order....\n')
            close_position(position)

def calculate_pnl():

    position = has_position()
    if position is None:
        print("No position found for ticket")
        return 0.0
    
    symbol = position.symbol
    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        print('Failed to get tick data')
        return 0.0
    
    entry_price = position.price_open
    current_price = tick.bid if position.type == 0 else tick.ask
    
    lot_size = position.volume
    contract_size = mt5.symbol_info(position.symbol).trade_contract_size

    if position.type == 0:  
        pnl = (current_price - entry_price) * lot_size * contract_size
    else:  
        pnl = (entry_price - current_price) * lot_size * contract_size

    return pnl