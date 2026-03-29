"""
APEX Signal Engine — Flask Web Uygulaması
Gerçek zamanlı Forex/Kripto binary sinyal sistemi
"""

import os
import json
import asyncio
import threading
import time
from datetime import datetime, timedelta
from collections import deque

import yfinance as yf
import pandas as pd
from flask import Flask, render_template, jsonify, request, session
from flask_cors import CORS

from apex_engine import analyze_apex, score_to_risk_level, RISK_THRESHOLDS

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "apex_secret_2024")
CORS(app)

BOT_PASSWORD = "free"

# ── Pariteler ─────────────────────────────────────────────────
FOREX_PAIRS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X", "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X", "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X", "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X", "GBP/JPY": "GBPJPY=X",
}
CRYPTO_PAIRS = {
    "BTC/USDT": "BTC-USD", "ETH/USDT": "ETH-USD",
    "XRP/USDT": "XRP-USD", "SOL/USDT": "SOL-USD",
    "BNB/USDT": "BNB-USD", "ADA/USDT": "ADA-USD",
    "DOGE/USDT": "DOGE-USD", "AVAX/USDT": "AVAX-USD",
    "DOT/USDT": "DOT-USD",  "LTC/USDT": "LTC-USD",
}
TIMEFRAMES = {
    "5m":  {"yf": "5m",  "period": "5d",  "expiry": "5 dk",   "resample": None,    "minutes": 5},
    "15m": {"yf": "15m", "period": "7d",  "expiry": "15 dk",  "resample": None,    "minutes": 15},
    "30m": {"yf": "30m", "period": "60d", "expiry": "30 dk",  "resample": None,    "minutes": 30},
    "45m": {"yf": "15m", "period": "7d",  "expiry": "45 dk",  "resample": "45min", "minutes": 45},
    "1h":  {"yf": "1h",  "period": "30d", "expiry": "1 Saat", "resample": None,    "minutes": 60},
}

# ── Global sinyal deposu ──────────────────────────────────────
signal_store   = deque(maxlen=100)   # Son 100 sinyal
scan_status    = {"running": False, "last_scan": None, "next_scan": None}
stats_store    = {"total": 0, "wins": 0, "losses": 0, "pending": 0}
pending_checks = []   # WIN/LOSS bekleyen sinyaller
_scan_lock     = threading.Lock()


def _get_ohlcv(yf_symbol, interval, period, resample=None):
    try:
        df = yf.download(yf_symbol, interval=interval, period=period,
                         progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 55:
            return None
        df.columns = [
            c[0].lower() if isinstance(c, tuple) else c.lower()
            for c in df.columns
        ]
        if resample:
            df = df.resample(resample).agg({
                "open": "first", "high": "max",
                "low": "min", "close": "last", "volume": "sum"
            }).dropna()
            if len(df) < 55:
                return None
        return df
    except Exception:
        return None


def _get_price(yf_symbol):
    try:
        t = yf.Ticker(yf_symbol)
        p = t.fast_info.last_price
        if p and float(p) > 0:
            return float(p)
    except Exception:
        pass
    try:
        df = yf.download(yf_symbol, period="1d", interval="1m",
                         progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            col = [c for c in df.columns if "lose" in str(c).lower()]
            if col:
                return float(df[col[0]].iloc[-1])
    except Exception:
        pass
    return None


def _price_fmt(pair_name, price):
    n = pair_name.upper()
    if "JPY" in n:         return f"{price:.3f}"
    if any(c in n for c in ["BTC", "ETH", "SOL", "BNB", "AVAX"]): return f"{price:.2f}"
    if any(c in n for c in ["XRP", "ADA", "DOT", "LTC", "DOGE"]): return f"{price:.4f}"
    return f"{price:.5f}"


def _scan_pair(pair_name, yf_symbol, tf_key, tf_cfg, pair_type, risk_filter="high"):
    """Tek bir pariteyi tara, sinyal varsa döndür."""
    df = _get_ohlcv(yf_symbol, tf_cfg["yf"], tf_cfg["period"], tf_cfg["resample"])
    if df is None:
        return None

    result    = analyze_apex(df)
    direction = result["direction"]
    score     = result["score"]

    if not result["signal_valid"] or direction is None:
        return None

    threshold = RISK_THRESHOLDS.get(risk_filter, 4)
    if score < threshold:
        return None

    price = _get_price(yf_symbol)
    now   = datetime.utcnow()

    signal = {
        "id":          f"{pair_name}_{tf_key}_{int(now.timestamp())}",
        "pair":        pair_name,
        "pair_type":   pair_type,
        "direction":   direction,
        "tf_key":      tf_key,
        "expiry":      tf_cfg["expiry"],
        "score":       score,
        "max_score":   result["max_score"],
        "confidence":  int(result["confidence"] * 100),
        "adx":         round(result["adx"], 1),
        "risk":        score_to_risk_level(score),
        "entry_price": _price_fmt(pair_name, price) if price else "—",
        "entry_raw":   price,
        "yf_symbol":   yf_symbol,
        "details":     [d for d in result["details"] if "✅" in d or "🔥" in d][:5],
        "layer_scores": result.get("layer_scores", {}),
        "timestamp":   now.isoformat(),
        "time_str":    now.strftime("%H:%M"),
        "status":      "active",   # active / win / loss
        "resolve_at":  (now + timedelta(minutes=tf_cfg["minutes"])).isoformat(),
        "exit_price":  None,
    }
    return signal


def run_scan(pairs_filter="all", tf_filter="all", risk_filter="high", pair_name_filter=None):
    """Tek seferlik tarama."""
    global scan_status, stats_store

    with _scan_lock:
        if scan_status["running"]:
            return
        scan_status["running"] = True

    try:
        scan_status["last_scan"] = datetime.utcnow().isoformat()

        pairs_to_scan = []
        if pairs_filter in ("all", "forex"):
            for pname, ysym in FOREX_PAIRS.items():
                if pair_name_filter and pname != pair_name_filter:
                    continue
                pairs_to_scan.append((pname, ysym, "forex"))
        if pairs_filter in ("all", "crypto"):
            for pname, ysym in CRYPTO_PAIRS.items():
                if pair_name_filter and pname != pair_name_filter:
                    continue
                pairs_to_scan.append((pname, ysym, "crypto"))

        tfs = {tf_filter: TIMEFRAMES[tf_filter]} if tf_filter != "all" else TIMEFRAMES

        for pname, ysym, ptype in pairs_to_scan:
            for tf_key, tf_cfg in tfs.items():
                sig = _scan_pair(pname, ysym, tf_key, tf_cfg, ptype, risk_filter)
                if sig:
                    existing = [s for s in signal_store
                                if s["pair"] == pname and s["tf_key"] == tf_key
                                and s["status"] == "active"]
                    if not existing:
                        signal_store.appendleft(sig)
                        stats_store["total"] += 1
                        stats_store["pending"] += 1
                        pending_checks.append(sig)

        scan_status["next_scan"] = (
            datetime.utcnow() + timedelta(minutes=5)
        ).isoformat()

    finally:
        scan_status["running"] = False


# Aktif tarama ayarları (son kullanıcı tercihleri)
_active_scan_config = {
    "active":      False,
    "pairs":       "all",
    "timeframe":   "all",
    "risk":        "high",
    "pair_name":   None,
    "interval":    300,   # saniye (5 dakika)
}

def _auto_scan_loop():
    """Sürekli tarama döngüsü — arka planda çalışır."""
    while True:
        if _active_scan_config["active"]:
            run_scan(
                _active_scan_config["pairs"],
                _active_scan_config["timeframe"],
                _active_scan_config["risk"],
                _active_scan_config["pair_name"],
            )
            check_results()
        time.sleep(_active_scan_config["interval"])

# Uygulama başlarken döngüyü başlat
_loop_thread = threading.Thread(target=_auto_scan_loop, daemon=True)
_loop_thread.start()


def check_results():
    """WIN/LOSS sonuçlarını kontrol et."""
    now = datetime.utcnow()
    resolved = []
    for sig in pending_checks:
        resolve_at = datetime.fromisoformat(sig["resolve_at"])
        if now < resolve_at:
            continue
        exit_price = _get_price(sig["yf_symbol"])
        if exit_price is None:
            continue

        entry = sig.get("entry_raw")
        if entry:
            win = (
                (sig["direction"] == "AL" and exit_price > entry) or
                (sig["direction"] == "SAT" and exit_price < entry)
            )
            sig["status"]     = "win" if win else "loss"
            sig["exit_price"] = _price_fmt(sig["pair"], exit_price)
            if win:
                stats_store["wins"] += 1
            else:
                stats_store["losses"] += 1
            stats_store["pending"] = max(0, stats_store["pending"] - 1)
        resolved.append(sig)

    for s in resolved:
        if s in pending_checks:
            pending_checks.remove(s)


# ── Flask Routes ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    if data.get("password", "").strip().lower() == BOT_PASSWORD:
        session["auth"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Yanlış şifre"}), 401


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/scan", methods=["POST"])
def start_scan():
    if not session.get("auth"):
        return jsonify({"error": "Giriş yapılmadı"}), 403
    data = request.get_json() or {}

    # Sürekli tarama config'ini güncelle
    _active_scan_config["active"]    = True
    _active_scan_config["pairs"]     = data.get("pairs", "all")
    _active_scan_config["timeframe"] = data.get("timeframe", "all")
    _active_scan_config["risk"]      = data.get("risk", "high")
    _active_scan_config["pair_name"] = data.get("pair_name", None)

    # Hemen bir tarama başlat
    t = threading.Thread(
        target=run_scan,
        args=(
            _active_scan_config["pairs"],
            _active_scan_config["timeframe"],
            _active_scan_config["risk"],
            _active_scan_config["pair_name"],
        ),
        daemon=True,
    )
    t.start()
    return jsonify({"ok": True, "message": "Sürekli tarama başlatıldı"})


@app.route("/api/stop", methods=["POST"])
def stop_scan():
    if not session.get("auth"):
        return jsonify({"error": "Giriş yapılmadı"}), 403
    _active_scan_config["active"] = False
    return jsonify({"ok": True, "message": "Tarama durduruldu"})


@app.route("/api/signals")
def get_signals():
    if not session.get("auth"):
        return jsonify({"error": "Giriş yapılmadı"}), 403
    check_results()
    return jsonify({
        "signals": list(signal_store),
        "stats":   stats_store,
        "status":  scan_status,
    })


@app.route("/api/clear", methods=["POST"])
def clear_signals():
    if not session.get("auth"):
        return jsonify({"error": "Giriş yapılmadı"}), 403
    signal_store.clear()
    pending_checks.clear()
    stats_store.update({"total": 0, "wins": 0, "losses": 0, "pending": 0})
    return jsonify({"ok": True})


@app.route("/api/pairs")
def get_pairs():
    return jsonify({
        "forex":  list(FOREX_PAIRS.keys()),
        "crypto": list(CRYPTO_PAIRS.keys()),
        "timeframes": list(TIMEFRAMES.keys()),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
 
