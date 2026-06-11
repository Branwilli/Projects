import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler

from model import PredictionModel, PAIRS, PAIR_TO_IDX
from indicators import add_features, add_targets, standardize_df, filter_data, smooth_labels
import warnings
import glob

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PAIR_CSV = {
    'EURUSD': 'data/DAT_MT_EURUSD_M1_*.csv',
    'GBPUSD': 'data/DAT_MT_GBPUSD_M1_*.csv',
    'USDCAD': 'data/DAT_MT_USDCAD_M1_*.csv',
    'AUDUSD': 'data/DAT_MT_AUDUSD_M1_*.csv',
}

HISTDATA_COLS = ['datetime', 'open', 'high', 'low', 'close', 'volume']

FEATURES = [
    'volatility', 'volatility_spike', 'volume_spike', 'range_expansion',
    'trend_strength', 'RSI', 'momentum_5', 'htf_bias', 'ma_10', 'ma_20',
    'vol_trend', 'expansion_strength', 'rsi_mom', 'trend_bias',
    'residuals', 'arima_pred'
]

WINDOW     = 60
EPOCHS     = 120
LR         = 0.0005
BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Data loading — one pipeline per pair
# ---------------------------------------------------------------------------
def load_pair(symbol, csv_glob):
    files = sorted(glob.glob(csv_glob))
 
    if not files:
        raise FileNotFoundError(
            f"No files found for {symbol} matching pattern: {csv_glob}\n"
            f"Check that your CSV files are in the right folder and the "
            f"PAIR_CSV pattern matches your filenames."
        )
 
    print(f"Loading {symbol} from {len(files)} file(s): {[os.path.basename(f) for f in files]}")
 
    frames = []
    for f in files:
        try:
            chunk = pd.read_csv(f, sep=',', header=None,
                                names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
        except Exception as e:
            print(f"Failed to read {f}: {e}")
            continue
 
        # Combine separate date and time columns into one datetime string
        chunk['datetime'] = chunk['date'].str.strip() + ' ' + chunk['time'].str.strip()
        chunk = chunk.drop(columns=['date', 'time'])
        frames.append(chunk)
 
    raw = pd.concat(frames).drop_duplicates(subset='datetime').sort_values('datetime')
    raw = raw.reset_index(drop=True)
    print(f"  [{symbol}] Raw rows after concat: {len(raw)}")
 
    # Combined format from Histdata MT: '2023.01.01 17:04'
    raw['datetime'] = pd.to_datetime(raw['datetime'], format='%Y.%m.%d %H:%M')
    raw = raw.dropna(subset=['datetime'])
    print(f"  [{symbol}] Rows after datetime parse: {len(raw)}")
 
    # Set index and resample M1 → 15-min directly.
    # Bypasses standardize_df() to avoid its 365-day trim which would cut
    # all historical data outside the last year.
    raw = raw.set_index('datetime').sort_index()
    raw = raw[['open', 'high', 'low', 'close', 'volume']]
    raw.columns = ['open', 'high', 'low', 'close', 'volume']
 
    df = raw.resample('15min').agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum'
    }).dropna()
    print(f"  [{symbol}] Rows after standardize_df: {len(df)}")
 
    df = add_features(df)
    print(f"  [{symbol}] Rows after add_features: {len(df)}")
 
    df = add_targets(df)
    print(f"  [{symbol}] Rows after add_targets + dropna: {len(df)}")
 
    df = filter_data(df).dropna()
    print(f"  [{symbol}] Rows after filter_data: {len(df)}")
 
    # Remove flat/repeated price sections that cause ARIMA Schur decomposition errors
    df = df[df['close'].diff().abs() > 1e-8].copy()
    df = df[df['close'].rolling(10).std() > 1e-6].copy()
    df = df.dropna()
 
    try:
        arima = ARIMA(df['close'], order=(2, 1, 2)).fit()
        df['arima_pred'] = arima.fittedvalues
        df['residuals']  = df['close'] - df['arima_pred']
    except Exception as e:
        print(f"[{symbol}] ARIMA failed ({e}) — using fallback (rolling mean)")
        df['arima_pred'] = df['close'].rolling(5, min_periods=1).mean()
        df['residuals']  = df['close'] - df['arima_pred']
    df = df.dropna()
 
    df['pair_id'] = PAIR_TO_IDX[symbol]
    print(f"{symbol}: {len(df)} rows after processing")
    return df
 
 
all_dfs = [load_pair(sym, path) for sym, path in PAIR_CSV.items()]
 
# ---------------------------------------------------------------------------
# Fit a single scaler across all pairs so features are on the same scale
# ---------------------------------------------------------------------------
combined = pd.concat(all_dfs)

scaler = StandardScaler()
scaler.fit(combined[FEATURES])   # fit once on all pairs combined

# ---------------------------------------------------------------------------
# Build sequences per pair then concatenate
# ---------------------------------------------------------------------------
def create_sequences(X, pair_ids, y1, y2, window=WINDOW):
    Xs, pids, y1s, y2s = [], [], [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i + window])
        pids.append(pair_ids[i + window])   # pair id at the label timestep
        y1s.append(y1[i + window])
        y2s.append(y2[i + window])
    return np.array(Xs), np.array(pids), np.array(y1s), np.array(y2s)


all_X, all_pids, all_y1, all_y2 = [], [], [], []

for df in all_dfs:
    X       = scaler.transform(df[FEATURES])
    pids    = df['pair_id'].values
    y1      = df['target_opportunity'].values
    y2      = np.nan_to_num(df['target_direction'].values, nan=0.0)

    Xs, ps, y1s, y2s = create_sequences(X, pids, y1, y2)
    all_X.append(Xs)
    all_pids.append(ps)
    all_y1.append(y1s)
    all_y2.append(y2s)

X_seq   = np.concatenate(all_X)
pid_seq = np.concatenate(all_pids)
y1_seq  = np.concatenate(all_y1)
y2_seq  = np.concatenate(all_y2)

# Shuffle so pairs are interleaved during training
rng  = np.random.default_rng(42)
perm = rng.permutation(len(X_seq))
X_seq, pid_seq, y1_seq, y2_seq = (
    X_seq[perm], pid_seq[perm], y1_seq[perm], y2_seq[perm]
)

split = int(len(X_seq) * 0.8)

X_train   = torch.tensor(X_seq[:split],   dtype=torch.float32)
pid_train = torch.tensor(pid_seq[:split], dtype=torch.long)
y1_train  = torch.tensor(y1_seq[:split],  dtype=torch.float32)
y2_train  = torch.tensor(y2_seq[:split],  dtype=torch.float32)

X_test    = torch.tensor(X_seq[split:],   dtype=torch.float32)
pid_test  = torch.tensor(pid_seq[split:], dtype=torch.long)
y1_test   = torch.tensor(y1_seq[split:],  dtype=torch.float32)

# ---------------------------------------------------------------------------
# Model, loss, optimizer
# ---------------------------------------------------------------------------
model = PredictionModel(input_size=len(FEATURES), embedding_dim=16).to(device)

pos_weight_opp = (len(y1_train) - y1_train.sum()) / (y1_train.sum() + 1e-9)
mask_dir       = y1_train > 0
pos_weight_dir = (
    (mask_dir.sum() - y2_train[mask_dir].sum()) /
    (y2_train[mask_dir].sum() + 1e-9)
)

criterion_opp = nn.BCEWithLogitsLoss(pos_weight=pos_weight_opp)
criterion_dir = nn.BCEWithLogitsLoss(pos_weight=pos_weight_dir)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5
)

# ---------------------------------------------------------------------------
# Training loop — mini-batch so all pairs are seen every epoch
# ---------------------------------------------------------------------------
n_train = len(X_train)

for epoch in range(EPOCHS):
    model.train()

    # Shuffle within epoch
    idx   = torch.randperm(n_train)
    epoch_loss, epoch_opp, epoch_dir = 0.0, 0.0, 0.0
    steps = 0

    for start in range(0, n_train, BATCH_SIZE):
        batch_idx = idx[start:start + BATCH_SIZE]

        xb   = X_train[batch_idx].to(device)
        pb   = pid_train[batch_idx].to(device)
        y1b  = y1_train[batch_idx].to(device)
        y2b  = y2_train[batch_idx].to(device)

        optimizer.zero_grad()
        opp_out, dir_out = model(xb, pb)
        opp_out = opp_out.squeeze()
        dir_out = dir_out.squeeze()

        # Opportunity loss with label smoothing
        y1_smooth  = smooth_labels(y1b, 0.08)
        loss_opp   = criterion_opp(opp_out, y1_smooth)

        # Direction loss — only on opportunity-positive samples
        opp_mask = y1b > 0
        if opp_mask.sum() > 0:
            y2_smooth = smooth_labels(y2b[opp_mask], 0.15)
            strength  = torch.abs(y2b[opp_mask] - 0.5) * 2
            loss_dir  = (criterion_dir(dir_out[opp_mask], y2_smooth) * strength).mean()
        else:
            loss_dir = torch.tensor(0.0, device=device)

        # Calibration penalty — push opp mean toward 0.40
        target_mean      = torch.tensor(0.40, device=device)
        calibration_loss = (torch.sigmoid(opp_out).mean() - target_mean) ** 2

        loss = 2.0 * loss_opp + 1.0 * loss_dir + 0.5 * calibration_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_opp  += loss_opp.item()
        epoch_dir  += loss_dir.item() if isinstance(loss_dir, torch.Tensor) else loss_dir
        steps      += 1

    avg_loss = epoch_loss / steps
    scheduler.step(avg_loss)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} "
        f"| Opp: {epoch_opp/steps:.4f} | Dir: {epoch_dir/steps:.4f}"
    )

# ---------------------------------------------------------------------------
# Evaluation — per-pair stats on test set
# ---------------------------------------------------------------------------
model.eval()

with torch.no_grad():
    opp_pred, dir_pred = model(X_test.to(device), pid_test.to(device))
    opp_pred = torch.sigmoid(opp_pred).cpu().numpy().flatten()
    dir_pred = torch.sigmoid(dir_pred).cpu().numpy().flatten()

print(f"\nOverall  | Opp mean: {opp_pred.mean():.3f} std: {opp_pred.std():.3f} "
      f"| Dir mean: {dir_pred.mean():.3f} std: {dir_pred.std():.3f}")

pid_test_np = pid_test.numpy()
for sym, idx in PAIR_TO_IDX.items():
    mask = pid_test_np == idx
    if mask.sum() == 0:
        continue
    print(
        f"{sym} | Opp mean: {opp_pred[mask].mean():.3f} "
        f"std: {opp_pred[mask].std():.3f} "
        f"| Dir mean: {dir_pred[mask].mean():.3f} "
        f"std: {dir_pred[mask].std():.3f}"
    )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs("saved_models", exist_ok=True)

torch.save({
    'model_state': model.state_dict(),
    'scaler':      scaler,
    'features':    FEATURES,
    'pairs':       PAIRS,
    'pair_to_idx': PAIR_TO_IDX,
}, "saved_models/multi_pair_lstm_model.pth")

print("\nModel saved to: saved_models/multi_pair_lstm_model.pth")