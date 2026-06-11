import os, time, torch, torch.nn as nn
import numpy as np, pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone

from trade import (conn, load_data, get_latest, has_position,
                   place_trade, close_position, recover_sl_pnl)
from indicators import standardize_df, add_features, filter_data, append_new_candle
from news import merge_df, sentiment_df, fetch_gdelt_news
from logs import log
from api import start_api, update_api_state, record_closed_trade
from model import PredictionModel, PAIRS, PAIR_TO_IDX
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings('ignore')


BUY  = 0
SELL = 1
HOLD = 2

CANDLE_MINUTES  = 15
ENTRY_WINDOW_S  = 10
MIN_TICK_WINDOW_S = 60
SAVE_PATH = "saved_models/multi_pair_lstm_model.pth"


# ---------------------------------------------------------------------------
# PPO heads — route through full attention path
# ---------------------------------------------------------------------------
class ValueHead(nn.Module):
    def __init__(self, base_model, hidden_size=128):
        super().__init__()
        self.base = base_model
        lstm_hidden = hidden_size
        for m in base_model.modules():
            if isinstance(m, nn.LSTM):
                lstm_hidden = m.hidden_size
                break
        self.value_head = nn.Linear(lstm_hidden, 1)

    def forward(self, x, pair_idx):
        out, _ = self.base.lstm(
            torch.cat([
                x,
                self.base.pair_embedding(pair_idx)
                    .unsqueeze(1).expand(-1, x.size(1), -1)
            ], dim=-1)
        )
        attn_out, _ = self.base.attention(out, out, out)
        h = self.base.attn_norm(out + attn_out).mean(dim=1)
        return self.value_head(h).squeeze(-1)


class PolicyHead(nn.Module):
    def __init__(self, base_model, hidden_size=128, n_actions=3):
        super().__init__()
        self.base = base_model
        lstm_hidden = hidden_size
        for m in base_model.modules():
            if isinstance(m, nn.LSTM):
                lstm_hidden = m.hidden_size
                break
        self.policy_head = nn.Linear(lstm_hidden, n_actions)

    def forward(self, x, pair_idx):
        out, _ = self.base.lstm(
            torch.cat([
                x,
                self.base.pair_embedding(pair_idx)
                    .unsqueeze(1).expand(-1, x.size(1), -1)
            ], dim=-1)
        )
        attn_out, _ = self.base.attention(out, out, out)
        h = self.base.attn_norm(out + attn_out).mean(dim=1)
        return self.policy_head(h)


class PPOActorCritic(nn.Module):
    def __init__(self, base_model, hidden_size=128, n_actions=3):
        super().__init__()
        self.base   = base_model
        self.policy = PolicyHead(base_model, hidden_size, n_actions)
        self.critic = ValueHead(base_model, hidden_size)

    def forward(self, x, pair_idx):
        opp, direction = self.base(x, pair_idx)
        opp       = torch.sigmoid(opp.squeeze(-1))
        direction = torch.sigmoid(direction.squeeze(-1))
        logits    = self.policy(x, pair_idx)
        value     = self.critic(x, pair_idx)
        return opp, direction, logits, value

    def act(self, x, pair_idx):
        opp, direction, logits, value = self.forward(x, pair_idx)
        dist     = torch.distributions.Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, opp, direction

    def evaluate(self, x, pair_idx, actions):
        _, _, logits, value = self.forward(x, pair_idx)
        dist     = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return log_prob, value, entropy


# ---------------------------------------------------------------------------
# Metrics — per pair
# ---------------------------------------------------------------------------
class Metrics:
    def __init__(self):
        self.trades = []
        self.wins   = []
        self.losses = []
        self.correct_direction = 0
        self.total_predictions = 0
        self.hold_count = 0

    def update(self, pnl, predicted_action):
        if pnl is None:
            return
        self.trades.append(pnl)
        if pnl > 0:
            self.wins.append(pnl)
        else:
            self.losses.append(pnl)
        actual    = 1 if pnl > 0 else 0
        predicted = 1 if predicted_action == BUY else 0
        if predicted == actual:
            self.correct_direction += 1
        self.total_predictions += 1

    def record_hold(self):
        self.hold_count += 1

    def summary(self):
        if not self.trades:
            return {}
        avg_win  = np.mean(self.wins)   if self.wins   else 0
        avg_loss = np.mean(self.losses) if self.losses else 0
        return {
            "Model Accuracy": self.correct_direction / max(self.total_predictions, 1),
            "Trade Accuracy": len(self.wins) / len(self.trades),
            "Avg Win":        avg_win,
            "Avg Loss":       avg_loss,
            "Win/Loss Ratio": abs(avg_win / avg_loss) if avg_loss != 0 else np.inf,
            "Sharpe Ratio":   (
                np.mean(self.trades) / (np.std(self.trades) + 1e-9)
            ) * np.sqrt(len(self.trades)) if len(self.trades) > 1 else 0,
            "Total Return":   np.sum(self.trades),
            "Hold Count":     self.hold_count,
        }


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------
def compute_reward(pnl):
    return np.tanh(pnl / 50.0)

def normalize_rewards(rewards):
    r = np.array(rewards, dtype=np.float32)
    return (r - r.mean()) / (r.std() + 1e-9)

def compute_discounted_rewards(rewards, gamma=0.99):
    discounted, R = [], 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.insert(0, R)
    return discounted


# ---------------------------------------------------------------------------
# PPO memory — shared across all pairs so the model learns jointly
# ---------------------------------------------------------------------------
class PPOMemory:
    def __init__(self):
        self.clear()

    def store(self, state, pair_idx, action, log_prob, value,
              reward, opp, bias, sentiment):
        self.states.append(state)
        self.pair_idxs.append(pair_idx)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.opps.append(opp)
        self.biases.append(bias)
        self.sentiments.append(sentiment)

    def store_pending(self, state, pair_idx, action, log_prob,
                      value, opp, bias, sentiment, symbol):
        self._pending[symbol] = dict(
            state=state, pair_idx=pair_idx, action=action,
            log_prob=log_prob, value=value, opp=opp,
            bias=bias, sentiment=sentiment
        )

    def commit_pending(self, reward, symbol):
        if self._pending is None:
            return
        p = self._pending[symbol]
        self.store(
            p['state'], p['pair_idx'], p['action'], p['log_prob'],
            p['value'], reward, p['opp'], p['bias'], p['sentiment']
        )
        self._pending[symbol] = None

    def get_pending(self, symbol):
        return self._pending.get(symbol, None)

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.states     = []
        self.pair_idxs  = []
        self.actions    = []
        self.log_probs  = []
        self.values     = []
        self.rewards    = []
        self.opps       = []
        self.biases     = []
        self.sentiments = []
        self._pending   = {}


# ---------------------------------------------------------------------------
# PPO update — pair_idx tensor passed to evaluate()
# ---------------------------------------------------------------------------
def ppo_update(
    actor_critic, optimizer, memory,
    batch_size=5, gamma=0.99, clip_eps=0.2,
    vf_coef=0.5, ent_coef=0.05, ppo_epochs=4,
):
    if len(memory) < batch_size:
        return

    raw_rewards  = memory.rewards[-batch_size:]
    disc_rewards = normalize_rewards(compute_discounted_rewards(raw_rewards, gamma))
    returns      = torch.tensor(disc_rewards, dtype=torch.float32)

    threshold  = 0.60
    advantages = []
    for opp, bias, sentiment, R in zip(
        memory.opps[-batch_size:],
        memory.biases[-batch_size:],
        memory.sentiments[-batch_size:],
        disc_rewards
    ):
        opp_weight     = max(0.01, 2.0 * (opp - threshold))
        context_weight = 1.0 + 0.2 * abs(bias) + 0.15 * abs(sentiment)
        advantages.append(np.clip(R * opp_weight * context_weight, -2.0, 2.0))

    advantages = torch.tensor(advantages, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

    states    = torch.tensor(
        np.array(memory.states[-batch_size:], dtype=np.float32)
    ).float()
    pair_idxs = torch.tensor(
        memory.pair_idxs[-batch_size:], dtype=torch.long
    )
    actions      = torch.tensor(memory.actions[-batch_size:],   dtype=torch.long)
    old_log_probs = torch.stack(memory.log_probs[-batch_size:]).detach()

    for _ in range(ppo_epochs):
        new_log_probs, values, entropy = actor_critic.evaluate(
            states, pair_idxs, actions
        )
        ratio       = torch.exp(new_log_probs - old_log_probs)
        surr1       = ratio * advantages
        surr2       = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss  = nn.functional.mse_loss(values, returns)
        entropy_loss = -entropy.mean()
        loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 1.0)
        optimizer.step()

    print(
        f'PPO | Policy: {policy_loss.item():.6f} '
        f'| Value: {value_loss.item():.6f} '
        f'| Entropy: {(-entropy_loss).item():.6f}'
    )
    pending_backup = memory._pending
    memory.clear()
    memory._pending = pending_backup

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_model_progress(actor_critic, features, scaler, pairs,
                        pair_to_idx, save_path=SAVE_PATH):
    torch.save({
        "model_state":       actor_critic.base.state_dict(),
        "policy_head_state": actor_critic.policy.policy_head.state_dict(),
        "value_head_state":  actor_critic.critic.value_head.state_dict(),
        "features":          features,
        "scaler":            scaler,
        "pairs":             pairs,
        "pair_to_idx":       pair_to_idx,
    }, save_path)
    print(f"Model saved to {save_path}")


def stop_training(actor_critic, features, scaler, pairs, pair_to_idx):
    print("Stopping live training...")
    save_model_progress(actor_critic, features, scaler, pairs, pair_to_idx)
    mt5.shutdown()
    print("MetaTrader connection closed.")


def load_existing_model(checkpoint_path=SAVE_PATH, hidden_size=128):
    ckpt        = torch.load(checkpoint_path, weights_only=False)
    features    = ckpt['features']
    scaler      = ckpt['scaler']
    pairs       = ckpt.get('pairs', PAIRS)
    pair_to_idx = ckpt.get('pair_to_idx', PAIR_TO_IDX)

    base_model  = PredictionModel(len(features))
    base_model.load_state_dict(ckpt['model_state'], strict=False)

    actor_critic = PPOActorCritic(base_model, hidden_size=hidden_size)
    if 'policy_head_state' in ckpt:
        actor_critic.policy.policy_head.load_state_dict(ckpt['policy_head_state'])
    if 'value_head_state' in ckpt:
        actor_critic.critic.value_head.load_state_dict(ckpt['value_head_state'])

    actor_critic.train()
    return actor_critic, features, scaler, pairs, pair_to_idx


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------
def arima_model(df):
    clean = df[df['close'].diff().abs() > 1e-8].copy()
    clean = clean[clean['close'].rolling(10).std() > 1e-6].copy()
 
    try:
        fitted           = ARIMA(clean['close'], order=(2, 1, 2)).fit()
        df['arima_pred'] = fitted.fittedvalues.reindex(df.index, method='ffill')
        df['residuals']  = df['close'] - df['arima_pred']
    except Exception as e:
        print(f"[ARIMA] Fit failed ({e}) — using rolling mean fallback")
        df['arima_pred'] = df['close'].rolling(5, min_periods=1).mean()
        df['residuals']  = df['close'] - df['arima_pred']
 
    return df


# ---------------------------------------------------------------------------
# Tick collection — symbol-aware, returns updated df for that symbol
# ---------------------------------------------------------------------------
def collect_ticks_until_open(symbol, df, interval_minutes=CANDLE_MINUTES):
    now                = datetime.now(timezone.utc)
    seconds_per_candle = interval_minutes * 60
    elapsed            = (now.minute % interval_minutes) * 60 + now.second + now.microsecond / 1e6
    wait_sec           = seconds_per_candle - elapsed
    candle_open_time   = now + timedelta(seconds=wait_sec)

    # If there isn't enough time to collect a meaningful window, just wait
    if wait_sec < MIN_TICK_WINDOW_S:
        print(f"[{symbol}] Only {wait_sec:.1f}s until candle — skipping tick collect.")
        time.sleep(wait_sec)
        return datetime.now(timezone.utc), df

    print(f"\n[{symbol}] Collecting ticks for {wait_sec:.1f}s...")

    tick_rows = []
    while datetime.now(timezone.utc) < candle_open_time:
        ticks = mt5.copy_ticks_from(
            symbol,
            datetime.now() - timedelta(seconds=1),
            1000, mt5.COPY_TICKS_ALL
        )
        if ticks is not None and len(ticks) > 0:
            tick_rows.append(pd.DataFrame(ticks))
        time.sleep(1)

    candle_open_time = datetime.now(timezone.utc)

    if not tick_rows:
        print(f"[{symbol}] No ticks — falling back to MT5 candle.")
        return candle_open_time, df

    all_ticks = (
        pd.concat(tick_rows)
        .drop_duplicates(subset='time')
        .sort_values('time')
    )
    mid = (all_ticks['ask'] + all_ticks['bid']) / 2

    raw = pd.DataFrame({
        'time':        all_ticks['time'].values,
        'open':        mid.values,
        'high':        mid.values,
        'low':         mid.values,
        'close':       mid.values,
        'tick_volume': all_ticks.get(
            'volume', pd.Series(1, index=all_ticks.index)
        ).values,
    })

    new_candle = standardize_df(raw)
    
    if new_candle.empty or new_candle.index[-1] in df.index:
        print(f"[{symbol}] Tick candle already in df or empty — skipping.")
        return candle_open_time, df

    df = append_new_candle(df, new_candle).tail(2000)
    print(
        f"[{symbol}] Tick candle appended | "
        f"O:{new_candle['open'].iloc[-1]:.5f} "
        f"H:{new_candle['high'].iloc[-1]:.5f} "
        f"L:{new_candle['low'].iloc[-1]:.5f} "
        f"C:{new_candle['close'].iloc[-1]:.5f}"
    )
    return candle_open_time, df


def within_entry_window(candle_open_time, window_seconds=ENTRY_WINDOW_S):
    elapsed = (datetime.now(timezone.utc) - candle_open_time).total_seconds()
    return elapsed <= window_seconds


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def live_training_loop(checkpoint_path=SAVE_PATH):
    conn()
    log('-------------- Live Training — Multi-Pair PPO ---------------')

    balance = mt5.account_info().balance

    # Load shared sentiment
    s_df = sentiment_df(fetch_gdelt_news())

    actor_critic, features, scaler, pairs, pair_to_idx = load_existing_model(
        checkpoint_path
    )

    # Freeze base model — only policy and value heads learn from live trades
    for param in actor_critic.base.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        list(actor_critic.policy.parameters()) +
        list(actor_critic.critic.parameters()),
        lr=3e-5
    )

    # Shared PPO memory — all pairs feed the same replay buffer
    memory = PPOMemory()

    pair_states = {}
    for symbol in pairs:
        df_sym = standardize_df(load_data(symbol))
        df_sym = add_features(df_sym)
        df_sym = filter_data(df_sym)
        df_sym = arima_model(df_sym)
        df_sym = merge_df(s_df, df_sym)
        pair_states[symbol] = {
            'df':          df_sym,
            'last_action': HOLD,
            'metrics':     Metrics(),
        }
        log(f"[{symbol}] Data loaded — {len(df_sym)} rows")
    
    all_live = pd.concat([
        pair_states[sym]['df'][features] for sym in pairs
    ]).dropna()
    scaler.fit(all_live)
    log("Scaler recalibrated on live data distribution")

    # Start dashboard API in background thread
    start_api(host="0.0.0.0", port=5050)

    try:
        while True:
            # Tick collection 
            first_symbol = pairs[0]
            candle_open_time, pair_states[first_symbol]['df'] = (
                collect_ticks_until_open(
                    first_symbol, pair_states[first_symbol]['df']
                )
            )

            # Remaining pairs — fetch closed candle from MT5 directly
            # (boundary already passed so ticks would be partial)
            for symbol in pairs[1:]:
                new = standardize_df(get_latest(symbol))
                
                if new.index[-1] not in pair_states[symbol]['df'].index:
                    pair_states[symbol]['df'] = append_new_candle(
                        pair_states[symbol]['df'], new
                    ).tail(2000)

            # ARIMA refit and cleanup for all pairs
            for symbol in pairs:
                pair_states[symbol]['df'] = arima_model(
                    pair_states[symbol]['df']
                ).dropna()

            entry_allowed = within_entry_window(candle_open_time)
            if not entry_allowed:
                print("[Missed Entry] Processing exceeded entry window.")

            
            # Per-pair forward pass, entry, and exit
            for symbol in pairs:
                pnl        = None    # ← reset every pair iteration
                close_type = None    # ← reset every pair iteration

                state   = pair_states[symbol]
                df_sym  = state['df']
                pid     = pair_to_idx[symbol]
                pid_t   = torch.tensor([pid], dtype=torch.long)

                # Build input tensor
                X        = scaler.transform(df_sym[features].values[-60:])
                X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

                # Forward pass
                with torch.no_grad():
                    action, log_prob, value, opp, direction = actor_critic.act(
                        X_tensor, pid_t
                    )

                action   = action.item()
                log_prob = log_prob.detach()
                value    = value.detach()
                opp_val  = opp.item()

                bias      = np.sign(df_sym['htf_bias_hourly'].iloc[-1])
                sentiment = df_sym['sentiment_score'].iloc[-1]
                position  = has_position(symbol)
                pending = memory.get_pending(symbol)

                label = {BUY: 'BUY', SELL: 'SELL', HOLD: 'HOLD'}[action]
                log(
                    f'[{symbol}] Opp: {opp_val:.4f} | Dir: {direction.item():.4f} '
                    f'| Bias: {bias} | Sentiment: {sentiment:.4f} | Action: {label}'
                    + ('' if entry_allowed else ' [SKIPPED]')
                )

                # ---- entry ----
                if (not position and pending is not None
                      and pending['pair_idx'] == pid):
                    pnl        = recover_sl_pnl(symbol)
                    close_type = 'SL/TP HIT'
                    print(f'[{symbol}] *** {close_type} ***')

                elif entry_allowed and not position and opp_val > 0.3:
                    if action == BUY:
                        place_trade('BUY', symbol, balance, 0.0001)
                        
                        state['last_action'] = BUY
                        memory.store_pending(
                            X, pid, action, log_prob, value,
                            opp_val, bias, sentiment, symbol
                        )
                    elif action == SELL:
                        place_trade('SELL', symbol, balance, 0.0001)
                        
                        state['last_action'] = SELL
                        memory.store_pending(
                            X, pid, action, log_prob, value,
                            opp_val, bias, sentiment, symbol
                        )
                    elif action == HOLD:
                        state['metrics'].record_hold()

                elif entry_allowed and action == HOLD and not position:
                    memory.store(
                        X, pid, action, log_prob, value,
                        -0.02, opp_val, bias, sentiment
                    )
                    state['metrics'].record_hold()

                # Re-check after potential entry
                #position = has_position(symbol)

                pending_snapshot = memory.get_pending(symbol)

                # ---- exit + RL update ----
                if position and pending is not None and pending['pair_idx'] == pid:
                    pnl        = close_position(position, symbol)
                    close_type = 'BOT CLOSE'

                if pnl is not None and pending_snapshot is not None:
                    base_reward    = compute_reward(pnl)
                    trade_dir      = 1 if pending_snapshot['action'] == BUY else -1
                    bias_align     = trade_dir * pending_snapshot['bias']
                    sent_align     = trade_dir * np.sign(pending_snapshot['sentiment'])
                    reward         = base_reward + abs(base_reward) * 0.25 * (
                        bias_align + sent_align
                    )

                    print(f'[{symbol}][{close_type}] PnL: {pnl:.5f} | Reward: {reward:.5f}')

                    tick = mt5.symbol_info_tick(symbol)
                    record_closed_trade(
                        symbol      = symbol,
                        action      = pending_snapshot['action'],
                        entry_price = pending_snapshot.get('entry_price', 0.0),
                        close_price = float(tick.bid) if tick else 0.0,
                        lots        = 1.0,
                        pnl         = pnl,
                        status      = 'success' if pnl > 0 else 'failed',
                        close_type  = close_type,
                    )

                    memory.commit_pending(reward, symbol)
                    ppo_update(actor_critic, optimizer, memory)

                    state['metrics'].update(pnl, state['last_action'])

                    if len(state['metrics'].trades) % 10 == 0:
                        stats = state['metrics'].summary()
                        log(f"""
                        === {symbol} PERFORMANCE ===
                        Model Accuracy:  {stats['Model Accuracy']:.3f}
                        Trade Accuracy:  {stats['Trade Accuracy']:.3f}
                        Avg Win:         {stats['Avg Win']:.5f}
                        Avg Loss:        {stats['Avg Loss']:.5f}
                        Win/Loss Ratio:  {stats['Win/Loss Ratio']:.3f}
                        Sharpe Ratio:    {stats['Sharpe Ratio']:.3f}
                        Total Return:    {stats['Total Return']:.5f}
                        Hold Count:      {stats['Hold Count']}
                        ============================
                        """)
 
            # Push latest state to dashboard API after all pairs processed
            update_api_state(pair_states, memory, pairs)

    except KeyboardInterrupt:
        stop_training(
            actor_critic, features, scaler, pairs, pair_to_idx
        )


if __name__ == '__main__':
    live_training_loop()