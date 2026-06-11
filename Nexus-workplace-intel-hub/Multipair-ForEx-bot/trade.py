import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from logs import log
import MetaTrader5 as mt5

load_dotenv()

LOGIN    = os.getenv('LOGIN')
PASSWORD = os.getenv('PASSWORD')
SERVER   = os.getenv('SERVER')

RISK_PER_TRADE = os.getenv('RISK_PER_TRADE')
SL_PIPS        = os.getenv('SL_PIPS')
TP_PIPS        = os.getenv('TP_PIPS')

# Pip value per standard lot, per symbol.
# All four pairs are USD-quoted or USD-base so value is 10 for USD accounts.
# Extend this dict for any future pairs with different pip values.
PIP_VALUES = {
    'EURUSD': 10,
    'GBPUSD': 10,
    'USDCAD': 10,
    'AUDUSD': 10,
}


def conn():
    if not mt5.initialize():
        raise RuntimeError('MT5 init failed')
    mt5.login(LOGIN, PASSWORD, SERVER)
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


def get_latest(symbol='EURUSD'):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 1, 3)
    if rates is None:
        print('MT5 Error: ', mt5.last_error())
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    if df.empty:
        print('No data returned')
    return df
    

def calculate_lot(symbol, balance, risk_pct, sl_pip, max_lot=1.0):
    pip_value  = PIP_VALUES.get(symbol, 10)
    risk_amount = balance * risk_pct
    lot = risk_amount / (sl_pip * pip_value)
    return round(min(max(lot, 0.01), max_lot), 2)


def has_position(symbol):
    """Return the first open position for symbol, or None."""
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        return positions[0]
    return None


def close_position(position, symbol):
    """Close position and return realised PnL."""
    pnl = calculate_pnl(symbol)

    order_type  = (
        mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY
        else mt5.ORDER_TYPE_BUY
    )
    symbol_info = mt5.symbol_info_tick(position.symbol)
    price       = symbol_info.bid if order_type == mt5.ORDER_TYPE_SELL else symbol_info.ask

    request = {
        'action':      mt5.TRADE_ACTION_DEAL,
        'symbol':      position.symbol,
        'volume':      position.volume,
        'type':        order_type,
        'position':    position.ticket,
        'price':       price,
        'deviation':   20,
        'magic':       123456,
        'comment':     'Close via Script',
        'type_time':   mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to close {position.ticket} ({symbol}): {result.comment}")

    log(f"[{symbol}] Position {position.ticket} closed | Lot: {position.volume}")
    return pnl


def place_trade(direction, symbol, balance, min_stop):
    lot     = calculate_lot(symbol, float(balance), float(RISK_PER_TRADE), float(SL_PIPS))
    tick    = mt5.symbol_info_tick(symbol)
    spread  = tick.ask - tick.bid
    sl_dist = max(min_stop, float(SL_PIPS) * 0.0001)
    tp_dist = max(min_stop, float(TP_PIPS) * 0.0001)

    if direction == 'BUY':
        price      = float(tick.ask)
        sl         = price - sl_dist
        tp         = price + tp_dist + spread  # Add spread to TP for buys
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price      = float(tick.bid)
        sl         = price + sl_dist + spread  # Add spread to SL for sells
        tp         = price - tp_dist
        order_type = mt5.ORDER_TYPE_SELL

    request = {
        'action':       mt5.TRADE_ACTION_DEAL,
        'symbol':       symbol,
        'volume':       lot,
        'type':         order_type,
        'price':        price,
        'sl':           sl,
        'tp':           tp,
        'deviation':    20,
        'magic':        123456,
        'comment':      'AI Trade',
        'type_time':    mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_FOK,
    }

    print(f'[{symbol}] Attempting to place {direction} trade...')
    result = mt5.order_send(request)

    if result is None:
        log(f'[{symbol}] Order failed: {mt5.last_error()}')
        return

    log(f"[{symbol}] TRADE {direction} | Lot: {lot} | Result: {result.retcode}")


def calculate_pnl(symbol):
    """Calculate unrealised PnL for the open position on symbol."""
    position = has_position(symbol)
    if position is None:
        return 0.0

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f'[{symbol}] Failed to get tick data')
        return 0.0

    entry_price    = position.price_open
    current_price  = tick.bid if position.type == 0 else tick.ask
    lot_size       = position.volume
    contract_size  = mt5.symbol_info(symbol).trade_contract_size

    if position.type == 0:
        pnl = (current_price - entry_price) * lot_size * contract_size
    else:
        pnl = (entry_price - current_price) * lot_size * contract_size

    return pnl


def recover_sl_pnl(symbol, lookback_hours=1):
    """Recover PnL from MT5 deal history when SL/TP closes a position."""
    from datetime import datetime, timedelta
    from_dt = datetime.now() - timedelta(hours=lookback_hours)
    deals   = mt5.history_deals_get(from_dt, datetime.now())

    if deals:
        closed = [
            d for d in deals
            if d.entry == mt5.DEAL_ENTRY_OUT and d.symbol == symbol
        ]
        if closed:
            return sorted(closed, key=lambda d: d.time)[-1].profit

    print(f"[{symbol}] SL/TP recovery: no closed deal found — using 0.0 fallback")
    return 0.0