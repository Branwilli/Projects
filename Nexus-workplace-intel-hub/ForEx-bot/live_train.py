import os, time, torch, torch.nn as nn
import numpy as np, pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone

from trade import conn, load_data, get_latest, has_position, place_trade, close_position
from indicators import standardize_df, add_features, filter_data, append_new_candle
from news import merge_df, sentiment_df, fetch_gdelt_news
from logs import log
from model import PredictionModel
from statsmodels.tsa.arima.model import ARIMA

import warnings

warnings.filterwarnings('ignore')

# --- Action constants ---
BUY  = 0
SELL = 1
HOLD = 2

# ---------------------------------------------------------------------------
# PPO Critic head
# A lightweight value network that shares the LSTM backbone of PredictionModel
# and adds a single linear head to estimate V(s).
# ---------------------------------------------------------------------------
class ValueHead(nn.Module):
    """Wraps an existing PredictionModel and adds a scalar value output."""
    def __init__(self, base_model: nn.Module, hidden_size: int = 128):
        super().__init__()
        self.base = base_model
        # Walk the base model to find its LSTM hidden size automatically
        lstm_hidden = hidden_size
        for m in base_model.modules():
            if isinstance(m, nn.LSTM):
                lstm_hidden = m.hidden_size
                break
        self.value_head = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        # Re-use the LSTM hidden state from the base model's lstm layer
        # We hook into the lstm output before the base model's own heads
        out, _ = self.base.lstm(x)          # (batch, seq, hidden)

        attn_out, _ = self.base.attention(out, out, out)
        h = self.base.attn_norm(out + attn_out).mean(dim=1)      

        value = self.value_head(h)          # (batch, 1)
        return value.squeeze(-1)            # (batch,)


# ---------------------------------------------------------------------------
# Three-way policy head
# Replaces the Bernoulli (Buy/Sell) with a Categorical (Buy/Sell/Hold).
# ---------------------------------------------------------------------------
class PolicyHead(nn.Module):
    """Maps the LSTM hidden state to logits over {BUY, SELL, HOLD}."""
    def __init__(self, base_model: nn.Module, hidden_size: int = 128, n_actions: int = 3):
        super().__init__()
        self.base = base_model
        lstm_hidden = hidden_size
        for m in base_model.modules():
            if isinstance(m, nn.LSTM):
                lstm_hidden = m.hidden_size
                break
        self.policy_head = nn.Linear(lstm_hidden, n_actions)

    def forward(self, x):
        out, _ = self.base.lstm(x)
        attn_out, _ = self.base.attention(out, out, out)
        h = self.base.attn_norm(out + attn_out).mean(dim=1)
        logits = self.policy_head(h)        # (batch, 3)
        return logits


# ---------------------------------------------------------------------------
# Combined actor-critic module
# Holds both the shared backbone, the original opp/direction heads,
# the new policy (actor) head, and the value (critic) head.
# ---------------------------------------------------------------------------
class PPOActorCritic(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_size: int = 128, n_actions: int = 3):
        super().__init__()
        self.base = base_model          # PredictionModel (opp + direction heads intact)
        self.policy = PolicyHead(base_model, hidden_size, n_actions)
        self.critic = ValueHead(base_model, hidden_size)

    def forward(self, x):
        """Returns (opp, direction_prob, action_logits, state_value)."""
        opp, direction = self.base(x)
        opp       = torch.sigmoid(opp.squeeze(-1))
        direction = torch.sigmoid(direction.squeeze(-1))
        logits    = self.policy(x)          # (batch, 3)
        value     = self.critic(x)          # (batch,)
        return opp, direction, logits, value

    def act(self, x):
        """Sample an action and return (action, log_prob, state_value, opp, direction)."""
        opp, direction, logits, value = self.forward(x)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, opp, direction

    def evaluate(self, x, actions):
        """For PPO update: returns (log_probs, state_values, entropy)."""
        _, _, logits, value = self.forward(x)
        dist     = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return log_prob, value, entropy


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class Metrics:
    def __init__(self):
        self.trades = []
        self.wins   = []
        self.losses = []
        self.correct_direction  = 0
        self.total_predictions  = 0
        self.hold_count         = 0

    def update(self, pnl, predicted_action):
        if pnl is None:
            return
        self.trades.append(pnl)
        if pnl > 0:
            self.wins.append(pnl)
        else:
            self.losses.append(pnl)

        actual_direction = 1 if pnl > 0 else 0
        # BUY=0 maps to winning when pnl>0; SELL=1 maps to winning when pnl<0
        predicted_direction = 1 if predicted_action == BUY else 0
        if predicted_direction == actual_direction:
            self.correct_direction += 1
        self.total_predictions += 1

    def record_hold(self):
        self.hold_count += 1

    def summary(self):
        if not self.trades:
            return {}
        avg_win  = np.mean(self.wins)   if self.wins   else 0
        avg_loss = np.mean(self.losses) if self.losses else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        trade_accuracy = len(self.wins) / len(self.trades)
        model_accuracy = (
            self.correct_direction / self.total_predictions
            if self.total_predictions > 0 else 0
        )
        sharpe = (
            np.mean(self.trades) / (np.std(self.trades) + 1e-9)
        ) * np.sqrt(len(self.trades)) if len(self.trades) > 1 else 0
        total_return = np.sum(self.trades)
        return {
            "Model Accuracy": model_accuracy,
            "Trade Accuracy": trade_accuracy,
            "Avg Win":        avg_win,
            "Avg Loss":       avg_loss,
            "Win/Loss Ratio": win_loss_ratio,
            "Sharpe Ratio":   sharpe,
            "Total Return":   total_return,
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
# PPO replay buffer
# ---------------------------------------------------------------------------
class PPOMemory:
    def __init__(self):
        self.clear()

    def store(self, state, action, log_prob, value, reward,
              opp, bias, sentiment):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.opps.append(opp)
        self.biases.append(bias)
        self.sentiments.append(sentiment)

    def store_pending(self, state, action, log_prob, value, opp, bias, sentiment):
        """Store a trade before its reward is known (filled in on close)."""
        self._pending = dict(
            state=state, action=action, log_prob=log_prob,
            value=value, opp=opp, bias=bias, sentiment=sentiment
        )

    def commit_pending(self, reward):
        """Attach reward and move pending entry into the main buffer."""
        if self._pending is None:
            return
        p = self._pending
        self.store(
            p['state'], p['action'], p['log_prob'], p['value'], reward,
            p['opp'], p['bias'], p['sentiment']
        )
        self._pending = None

    def get_pending(self):
        return self._pending

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.states     = []
        self.actions    = []
        self.log_probs  = []
        self.values     = []
        self.rewards    = []
        self.opps       = []
        self.biases     = []
        self.sentiments = []
        self._pending   = None


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------
def ppo_update(
    actor_critic: PPOActorCritic,
    optimizer: torch.optim.Optimizer,
    memory: PPOMemory,
    batch_size: int   = 5,
    gamma: float      = 0.99,
    clip_eps: float   = 0.2,
    vf_coef: float    = 0.5,
    ent_coef: float   = 0.05,
    ppo_epochs: int   = 4,
):
    if len(memory) < batch_size:
        return

    # ---- build tensors ----
    raw_rewards = memory.rewards[-batch_size:]
    disc_rewards = normalize_rewards(compute_discounted_rewards(raw_rewards, gamma))
    returns = torch.tensor(disc_rewards, dtype=torch.float32)

    # Context-weighted advantages (preserving your bias/sentiment logic)
    threshold = 0.60
    advantages = []
    for i, (opp, bias, sentiment, R) in enumerate(
        zip(memory.opps[-batch_size:],
            memory.biases[-batch_size:],
            memory.sentiments[-batch_size:],
            disc_rewards)
    ):
        opp_weight       = max(0.01, 2.0 * (opp - threshold))
        context_weight   = 1.0 + 0.2 * abs(bias) + 0.15 * abs(sentiment)
        adv              = R * opp_weight * context_weight
        advantages.append(np.clip(adv, -2.0, 2.0))
    advantages = torch.tensor(advantages, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

    states_np = np.array(memory.states[-batch_size:], dtype=np.float32)
    states    = torch.tensor(states_np).float()           # (B, seq, features)
    actions   = torch.tensor(memory.actions[-batch_size:], dtype=torch.long)
    old_log_probs = torch.stack(memory.log_probs[-batch_size:]).detach()

    # ---- PPO epochs ----
    for _ in range(ppo_epochs):
        new_log_probs, values, entropy = actor_critic.evaluate(states, actions)

        # Clipped surrogate objective
        ratio        = torch.exp(new_log_probs - old_log_probs)
        surr1        = ratio * advantages
        surr2        = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss  = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss   = nn.functional.mse_loss(values, returns)

        # Entropy bonus (encourages exploration across all 3 actions)
        entropy_loss = -entropy.mean()

        loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 1.0)
        optimizer.step()

    print(
        f'PPO Update | Policy: {policy_loss.item():.6f} '
        f'| Value: {value_loss.item():.6f} '
        f'| Entropy: {(-entropy_loss).item():.6f}'
    )
    memory.clear()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
SAVE_PATH = "saved_models/2nd_hybrid_eurusd_lstm_model.pth"

def save_model_progress(actor_critic, features, scaler,
                        save_path=SAVE_PATH):
    checkpoint = {
        "model_state":        actor_critic.base.state_dict(),
        "policy_head_state":  actor_critic.policy.policy_head.state_dict(),
        "value_head_state":   actor_critic.critic.value_head.state_dict(),
        "features":           features,
        "scaler":             scaler,
    }
    torch.save(checkpoint, save_path)
    print(f"Model progress saved to {save_path}")

def stop_training(actor_critic, features, scaler):
    print("Stopping live training...")
    save_model_progress(actor_critic, features, scaler)
    mt5.shutdown()
    print("MetaTrader connection closed. Training stopped.")

def load_existing_model(checkpoint_path=SAVE_PATH, hidden_size=128):
    checkpoints  = torch.load(checkpoint_path, weights_only=False)
    features     = checkpoints['features']
    scaler       = checkpoints['scaler']
    model_state  = checkpoints['model_state']

    base_model = PredictionModel(len(features))
    base_model.load_state_dict(model_state)

    actor_critic = PPOActorCritic(base_model, hidden_size=hidden_size)

    # Restore policy/value heads if they were saved previously
    if 'policy_head_state' in checkpoints:
        actor_critic.policy.policy_head.load_state_dict(
            checkpoints['policy_head_state']
        )
    if 'value_head_state' in checkpoints:
        actor_critic.critic.value_head.load_state_dict(
            checkpoints['value_head_state']
        )

    actor_critic.train()
    return actor_critic, features, scaler


# ---------------------------------------------------------------------------
# ARIMA residual features
# ---------------------------------------------------------------------------
def arima_model(df):
    fitted = ARIMA(df['close'], order=(2, 1, 2)).fit()
    df['arima_pred'] = fitted.fittedvalues
    df['residuals']  = df['close'] - df['arima_pred']
    return df

def synchronize_entry_exit(df, interval_minutes=15, threshold_seconds=5):
    now = datetime.now(timezone.utc)
    minutes_into_candle = now.minute % interval_minutes
    seconds_to_wait = ((interval_minutes - minutes_into_candle) * 60) - now.second - now.microsecond / 1e6

    max_possible_wait = interval_minutes * 60

    if (seconds_to_wait > threshold_seconds and seconds_to_wait < (max_possible_wait - threshold_seconds)):
        print(f"\n[Time Sync] Synchronizing... Waiting for {seconds_to_wait:.2f}s until next candle.\n")

        tick_rows = []

        while datetime.now(timezone.utc) < now + timedelta(seconds=seconds_to_wait):
            from_dt = datetime.now() - timedelta(seconds=1)
            ticks = mt5.copy_ticks_from("EURUSD", from_dt, 1000, mt5.COPY_TICKS_ALL)

            if ticks is not None and len(ticks) > 0:
                tick_df = pd.DataFrame(ticks)
                tick_rows.append(tick_df)
            time.sleep(1)
        
        if not tick_rows:
            print('[Tick Collection] No ticks collected during synchronization window.\n')
        
        all_ticks = pd.concat(tick_rows).drop_duplicates(subset='time').sort_values('time')
        mid = (all_ticks['ask'] + all_ticks['bid']) / 2

        raw = pd.DataFrame({
            "time": all_ticks['time'].values,
            "open": mid.values,
            "high": mid.values,
            "low": mid.values,
            "close": mid.values,
            "volume": all_ticks.get('volume', pd.Series(1, index=all_ticks.index)).values
        })
        
        new_candle = standardize_df(raw)
       
        if new_candle.empty or new_candle.index[-1] in df.index:
            #print('[Tick Collection] No new candle formed during synchronization window.\n')
            return df
        
        #print(f'[Tick Collection] Collected {len(all_ticks)} ticks during synchronization window.\n')
        df = append_new_candle(df, new_candle).tail(2000)
        return df
    
    else:
        return 
        #print('\n[Time Sync] At boundary or within guard threshold window. Proceeding instantly\n') 


# ---------------------------------------------------------------------------
# SL/TP recovery — fetch actual PnL from MT5 deal history when a position
# is closed externally (stop-loss or take-profit hit) before the bot calls
# close_position(). Falls back to a worst-case SL estimate if no deal found.
# ---------------------------------------------------------------------------
def recover_sl_pnl(lookback_minutes: int = 15) -> float:
    """Return the PnL of the most recent closed deal from MT5 history."""
    now = datetime.now(timezone.utc)
    lookback_minutes = 15

    minutes_into_candle = now.minute % lookback_minutes
    candle_start = now.replace(second=0, microsecond=0) - timedelta(minutes=minutes_into_candle)
    prev_candle_start = candle_start - timedelta(minutes=lookback_minutes)
    prev_candle_end = candle_start

    deals   = mt5.history_deals_get(prev_candle_start, prev_candle_end)

    if deals:
        # Filter to actual entry/exit deals (DEAL_ENTRY_OUT = 1)
        closed = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
        if closed:
            last = sorted(closed, key=lambda d: d.time)[-1]
            print('SL/TP recovery: found closed deal in history with PnL')
            return last.profit

    # Fallback: return 0 so learning still triggers with a neutral signal
    print("SL/TP recovery: no closed deal found in history — using 0.0 fallback")
    return 0.0


# ---------------------------------------------------------------------------
# Main live PPO training loop
# ---------------------------------------------------------------------------
def live_training_loop(checkpoint_path=SAVE_PATH):
    conn()

    log('--------------------- Live Training (PPO) ----------------------------')
    account_info = mt5.account_info()
    balance      = account_info.balance

    s_df = sentiment_df(fetch_gdelt_news())

    df = standardize_df(load_data())
    df = add_features(df)
    df = filter_data(df)
    df = arima_model(df)
    df = merge_df(s_df, df)

    actor_critic, features, scaler = load_existing_model(checkpoint_path)

    for param in actor_critic.base.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        list(actor_critic.policy.parameters()) +
        list(actor_critic.critic.parameters()), 
        lr=3e-5
        )
    
    memory    = PPOMemory()
    metrics   = Metrics()

    last_action = HOLD   # track most recent non-hold action for metrics

    try:
        while True:
            synchronize_entry_exit(df)  # Ensure we start at the beginning of a new candle
            new = standardize_df(get_latest())

            pnl = None

            if new.index[-1] not in df.index:
                df = append_new_candle(df, new).tail(2000)
                df = arima_model(df)
                df = df.dropna()

                # ---- build input tensor ----
                X = df[features].values[-60:]
                X = scaler.transform(X)
                X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

                # ---- forward pass ----
                with torch.no_grad():
                    action, log_prob, value, opp, direction = actor_critic.act(X_tensor)

                action   = action.item()
                log_prob = log_prob.detach()
                value    = value.detach()
                opp_val  = opp.item()

                bias      = np.sign(df['htf_bias_hourly'].iloc[-1])
                sentiment = df['sentiment_score'].iloc[-1]
                position  = has_position()

                action_label = {BUY: 'BUY', SELL: 'SELL', HOLD: 'HOLD'}[action]
                log(
                    f'\nOpp: {opp_val:.4f} | Dir: {direction.item():.4f} '
                    f'| Bias: {bias} | Sentiment: {sentiment:.4f} '
                    f'| Action: {action_label}'
                )

                # ---- entry logic ----
                if not position and memory.get_pending() is not None:
                    print('\n*** SL/TP HIT detected — recovering PnL from MT5 history ***')
                    pnl = recover_sl_pnl()
                    
                elif not position and opp_val > 0.6:
                    if action == BUY:
                        print('\nPlacing BUY order')
                        place_trade('BUY', balance, 0.0001)
                        last_action = BUY
                        memory.store_pending(X, action, log_prob, value,
                                             opp_val, bias, sentiment)

                    elif action == SELL:
                        print('\nPlacing SELL order')
                        place_trade('SELL', balance, 0.0001)
                        last_action = SELL
                        memory.store_pending(X, action, log_prob, value,
                                             opp_val, bias, sentiment)

                    elif action == HOLD:
                        print('\nHOLD — no trade placed')
                        metrics.record_hold()

                elif action == HOLD and not position:
                    # Hold with no position: small negative reward to prevent
                    # the agent from learning to always hold (lazy policy)
                    hold_reward = -0.02
                    memory.store(X, action, log_prob, value, hold_reward,
                                 opp_val, bias, sentiment)
                    metrics.record_hold()

                # ---- exit + RL update ----
                if position and memory.get_pending() is not None:
                    pnl = close_position(position)

                if pnl is not None:
                    base_reward = compute_reward(pnl)

                    pending      = memory.get_pending()
                    trade_dir    = 1 if pending['action'] == BUY else -1
                    bias_align   = trade_dir * pending['bias']
                    sent_align   = trade_dir * np.sign(pending['sentiment'])

                    bonus_weight        = 0.25
                    alignment_modifier  = (
                        abs(base_reward) * bonus_weight * (bias_align + sent_align)
                    )
                    reward = base_reward + alignment_modifier

                    print(f'PnL: {pnl:.5f} | Reward: {reward:.5f}')

                    memory.commit_pending(reward)
                    ppo_update(actor_critic, optimizer, memory)

                    metrics.update(pnl, last_action)

                    if len(metrics.trades) % 10 == 0:
                        stats = metrics.summary()
                        log(f"""
                        ================ PERFORMANCE ================
                        Model Accuracy:  {stats['Model Accuracy']:.3f}
                        Trade Accuracy:  {stats['Trade Accuracy']:.3f}
                        Avg Win:         {stats['Avg Win']:.5f}
                        Avg Loss:        {stats['Avg Loss']:.5f}
                        Win/Loss Ratio:  {stats['Win/Loss Ratio']:.3f}
                        Sharpe Ratio:    {stats['Sharpe Ratio']:.3f}
                        Total Return:    {stats['Total Return']:.5f}
                        Hold Count:      {stats['Hold Count']}
                        ============================================
                        """)

            #time.sleep(900)

    except KeyboardInterrupt:
        stop_training(actor_critic, features, scaler)


if __name__ == '__main__':
    live_training_loop()
