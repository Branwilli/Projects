import sys
import subprocess
import threading
import signal
import os
from datetime import datetime, timezone
from flask import Flask, jsonify, request
from flask_cors import CORS
 
# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_state_lock      = threading.Lock()
_dashboard_state = {}
_trade_history   = {}
 
_bot_process     = None          # subprocess.Popen handle
_bot_lock        = threading.Lock()
_bot_log_buffer  = []            # last 100 lines of bot stdout
_bot_started_at  = None
_bot_stopped_at  = None
 
BOT_SCRIPT = os.path.join(os.path.dirname(__file__), "live_train.py")
 
 
app = Flask(__name__)
CORS(app)
 
 
@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    with _state_lock:
        snap = dict(_dashboard_state)
        snap["bot_status"] = _get_bot_status()
        # Merge closed trade history into each pair's data
        if "pairs" in snap:
            for symbol, pair_data in snap["pairs"].items():
                pair_data["trades"] = _trade_history.get(symbol, [])[-20:]
        return jsonify(snap)
 
 
@app.route("/api/history/<symbol>", methods=["GET"])
def history(symbol):
    with _state_lock:
        return jsonify(_trade_history.get(symbol.upper(), []))
 
 
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "ok",
        "botStatus": _get_bot_status(),
        "ts":        datetime.now(timezone.utc).isoformat(),
    })
 
 
@app.route("/api/bot/status", methods=["GET"])
def bot_status():
    return jsonify({
        "status":    _get_bot_status(),
        "pid":       _bot_process.pid if _bot_process else None,
        "startedAt": _bot_started_at.isoformat() if _bot_started_at else None,
        "stoppedAt": _bot_stopped_at.isoformat() if _bot_stopped_at else None,
        "logs":      _bot_log_buffer[-50:],   # last 50 log lines
    })
 
 
@app.route("/api/bot/start", methods=["POST"])
def bot_start():
    global _bot_process, _bot_started_at, _bot_stopped_at
 
    with _bot_lock:
        if _bot_process is not None and _bot_process.poll() is None:
            return jsonify({"ok": False, "error": "Bot is already running"}), 400
 
        try:
            _bot_process = subprocess.Popen(
                [sys.executable, BOT_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            _bot_started_at = datetime.now(timezone.utc)
            _bot_stopped_at = None
 
            # Stream stdout into log buffer in a background thread
            threading.Thread(
                target=_stream_bot_logs,
                args=(_bot_process,),
                daemon=True,
                name="bot-log-reader",
            ).start()
 
            return jsonify({
                "ok":  True,
                "pid": _bot_process.pid,
                "startedAt": _bot_started_at.isoformat(),
            })
 
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
 
 
@app.route("/api/bot/stop", methods=["POST"])
def bot_stop():
    global _bot_process, _bot_stopped_at
 
    with _bot_lock:
        if _bot_process is None or _bot_process.poll() is not None:
            return jsonify({"ok": False, "error": "Bot is not running"}), 400
 
        try:
            # Send SIGINT so live_train.py hits KeyboardInterrupt → saves model
            _bot_process.send_signal(signal.SIGINT)
            try:
                _bot_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                _bot_process.kill()
 
            _bot_stopped_at = datetime.now(timezone.utc)
            return jsonify({
                "ok":        True,
                "stoppedAt": _bot_stopped_at.isoformat(),
            })
 
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
 
 
@app.route("/api/bot/logs", methods=["GET"])
def bot_logs():
    """Return the last N lines of bot stdout. ?lines=100"""
    n = int(request.args.get("lines", 100))
    with _bot_lock:
        return jsonify({"logs": _bot_log_buffer[-n:]})
 
 
@app.route("/api/internal/state", methods=["POST"])
def internal_state():
    """Receive full dashboard snapshot from live_train.py subprocess."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"ok": False}), 400
    with _state_lock:
        _dashboard_state.update(data)
    return jsonify({"ok": True})
 
 
@app.route("/api/internal/trade", methods=["POST"])
def internal_trade():
    """Receive a closed trade record from live_train.py subprocess."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"ok": False}), 400
    symbol = data.get("symbol", "").upper()
    with _state_lock:
        if symbol not in _trade_history:
            _trade_history[symbol] = []
        _trade_history[symbol].append(data)
    return jsonify({"ok": True})
 
 
def update_api_state(pair_states, memory, pairs, api_url="http://localhost:5050"):
    """Build dashboard snapshot and POST it to the API server."""
    import requests
 
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pairs":     {},
        "totals":    {},
    }
 
    total_pnl    = 0.0
    total_wins   = 0
    total_losses = 0
    total_trades = 0
 
    for symbol in pairs:
        state   = pair_states[symbol]
        metrics = state["metrics"]
        pending = memory.get_pending(symbol)
 
        active_trade = None
        if pending is not None:
            action_map  = {0: "BUY", 1: "SELL", 2: "HOLD"}
            candle_time = pending.get("candle_time")
            active_trade = {
                "type":      action_map.get(pending["action"], "HOLD"),
                "lots":      1.0,
                "price":     float(pending.get("entry_price", 0.0)),
                "opp":       round(float(pending.get("opp", 0.0)), 4),
                "bias":      int(pending.get("bias", 0)),
                "sentiment": round(float(pending.get("sentiment", 0.0)), 4),
                "status":    "open",
                "time":      candle_time.strftime("%H:%M UTC") if candle_time else "--:-- UTC",
            }
 
        pair_pnl      = float(sum(metrics.trades)) if metrics.trades else 0.0
        total_pnl    += pair_pnl
        wins          = len(metrics.wins)
        losses        = len(metrics.losses)
        total_trades += wins + losses
        total_wins   += wins
        total_losses += losses
 
        summary = metrics.summary() if metrics.trades else {}
        for k, v in list(summary.items()):
            if isinstance(v, float) and (v != v or v == float("inf")):
                summary[k] = None
 
        snapshot["pairs"][symbol] = {
            "pnl":         round(pair_pnl, 2),
            "activeTrade": active_trade,
            "metrics":     summary,
            "holdCount":   metrics.hold_count,
        }
 
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    snapshot["totals"] = {
        "totalPnl":    round(total_pnl, 2),
        "totalTrades": total_trades,
        "wins":        total_wins,
        "losses":      total_losses,
        "winRate":     round(win_rate, 1),
        "activePairs": sum(1 for s in pairs if memory.get_pending(s) is not None),
    }
 
    try:
        requests.post(f"{api_url}/api/internal/state", json=snapshot, timeout=2)
    except Exception:
        pass   # never let a failed API push crash the trading loop
 
 
def record_closed_trade(symbol, action, entry_price, close_price,
                        lots, pnl, status, close_type,
                        api_url="http://localhost:5050"):
    """POST a closed trade record to the API server."""
    import requests
 
    action_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
    record = {
        "symbol":     symbol,
        "type":       action_map.get(action, "HOLD"),
        "lots":       lots,
        "entryPrice": round(float(entry_price), 5),
        "closePrice": round(float(close_price), 5),
        "pnl":        round(float(pnl), 2),
        "status":     status,
        "closeType":  close_type,
        "time":       datetime.now(timezone.utc).strftime("%H:%M UTC"),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }
 
    try:
        requests.post(f"{api_url}/api/internal/trade", json=record, timeout=2)
    except Exception:
        pass
 
 
def _get_bot_status():
    with _bot_lock:
        if _bot_process is None:
            return "STOPPED"
        if _bot_process.poll() is None:
            return "RUNNING"
        return "STOPPED"
 
 
def _stream_bot_logs(proc):
    """Read bot subprocess stdout line by line into the log buffer."""
    global _bot_stopped_at
    for line in proc.stdout:
        line = line.rstrip()
        print(f"[BOT] {line}")
        with _bot_lock:
            _bot_log_buffer.append(line)
            if len(_bot_log_buffer) > 500:
                _bot_log_buffer.pop(0)
    # Process ended
    with _bot_lock:
        _bot_stopped_at = datetime.now(timezone.utc)
 
 
def start_api(host="0.0.0.0", port=5050):
    """Start the Flask API (called from live_train.py when running embedded)."""
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    threading.Thread(
        target=lambda: app.run(host=host, port=port, use_reloader=False),
        daemon=True, name="dashboard-api",
    ).start()
    print(f"[API] Dashboard API running at http://{host}:{port}/api/dashboard")
 
 
if __name__ == "__main__":
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print("[API] Starting dashboard server on http://0.0.0.0:5050")
    print(f"[API] Bot script: {BOT_SCRIPT}")
    print("[API] Open the dashboard and click START to launch the bot")
    app.run(host="0.0.0.0", port=5050, use_reloader=False)