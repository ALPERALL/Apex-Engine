"""
APEX Signal Engine — Flask Web Uygulaması v3.0
"""
import os, json, threading, time, uuid, hashlib
from datetime import datetime, timedelta
from collections import deque

import yfinance as yf
import pandas as pd
from flask import Flask, render_template, jsonify, request, session
from flask_cors import CORS
from apex_engine import analyze_apex, score_to_risk_level, RISK_THRESHOLDS

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "apex_2024_secret")
CORS(app, supports_credentials=True)

FOREX_PAIRS = {
    "EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","USD/JPY":"USDJPY=X",
    "AUD/USD":"AUDUSD=X","USD/CAD":"USDCAD=X","USD/CHF":"USDCHF=X",
    "NZD/USD":"NZDUSD=X","EUR/GBP":"EURGBP=X","EUR/JPY":"EURJPY=X","GBP/JPY":"GBPJPY=X",
}
CRYPTO_PAIRS = {
    "BTC/USDT":"BTC-USD","ETH/USDT":"ETH-USD","XRP/USDT":"XRP-USD",
    "SOL/USDT":"SOL-USD","BNB/USDT":"BNB-USD","ADA/USDT":"ADA-USD",
    "DOGE/USDT":"DOGE-USD","AVAX/USDT":"AVAX-USD","DOT/USDT":"DOT-USD","LTC/USDT":"LTC-USD",
}
TIMEFRAMES = {
    "5m":  {"yf":"5m",  "period":"5d",  "expiry":"5 dk",   "resample":None,    "minutes":5},
    "15m": {"yf":"15m", "period":"7d",  "expiry":"15 dk",  "resample":None,    "minutes":15},
    "30m": {"yf":"30m", "period":"60d", "expiry":"30 dk",  "resample":None,    "minutes":30},
    "45m": {"yf":"15m", "period":"7d",  "expiry":"45 dk",  "resample":"45min", "minutes":45},
    "1h":  {"yf":"1h",  "period":"30d", "expiry":"1 Saat", "resample":None,    "minutes":60},
}

_users_db: dict = {}
_lock = threading.Lock()

def _hash(pw): return hashlib.sha256(pw.encode()).hexdigest()

signal_store   = deque(maxlen=200)
pending_checks = []
stats_store    = {"total":0,"wins":0,"losses":0,"pending":0}
scan_status    = {"running":False,"last_scan":None,"scanning_pair":None}
_scan_lock     = threading.Lock()
_active_cfg    = {"active":False,"pairs":"all","timeframe":"all","risk":"high","pair_name":None,"interval":300}

def _get_ohlcv(sym, interval, period, resample=None):
    try:
        df = yf.download(sym, interval=interval, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 55: return None
        df.columns = [c[0].lower() if isinstance(c,tuple) else c.lower() for c in df.columns]
        if resample:
            df = df.resample(resample).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
            if len(df) < 55: return None
        return df
    except Exception as e:
        app.logger.warning(f"OHLCV {sym}: {e}"); return None

def _get_price(sym):
    try:
        p = yf.Ticker(sym).fast_info.last_price
        if p and float(p)>0: return float(p)
    except: pass
    try:
        df = yf.download(sym, period="1d", interval="1m", progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            col = [c for c in df.columns if "lose" in str(c).lower()]
            if col: return float(df[col[0]].iloc[-1])
    except: pass
    return None

def _fmt(pair, price):
    n = pair.upper()
    if "JPY" in n: return f"{price:.3f}"
    if any(c in n for c in ["BTC","ETH","SOL","BNB","AVAX"]): return f"{price:.2f}"
    if any(c in n for c in ["XRP","ADA","DOT","LTC","DOGE"]): return f"{price:.4f}"
    return f"{price:.5f}"

def _scan_pair(pname, ysym, tf_key, tf_cfg, ptype, risk):
    df = _get_ohlcv(ysym, tf_cfg["yf"], tf_cfg["period"], tf_cfg["resample"])
    if df is None: return None
    result = analyze_apex(df)
    if not result["signal_valid"] or result["direction"] is None: return None
    if result["score"] < RISK_THRESHOLDS.get(risk, 4): return None
    price = _get_price(ysym)
    now   = datetime.utcnow()
    return {
        "id":          str(uuid.uuid4())[:8],
        "pair":        pname, "pair_type":ptype,
        "direction":   result["direction"],
        "tf_key":      tf_key, "expiry":tf_cfg["expiry"], "minutes":tf_cfg["minutes"],
        "score":       result["score"], "max_score":result["max_score"],
        "confidence":  int(result["confidence"]*100),
        "adx":         round(result["adx"],1),
        "risk":        score_to_risk_level(result["score"]),
        "entry_price": _fmt(pname,price) if price else "—",
        "entry_raw":   price, "yf_symbol":ysym,
        "details":     result["details"],
        "layer_scores":result.get("layer_scores",{}),
        "timestamp":   now.isoformat(), "time_str":now.strftime("%H:%M"),
        "status":      "active",
        "resolve_at":  (now+timedelta(minutes=tf_cfg["minutes"])).isoformat(),
        "exit_price":  None, "pnl":None,
    }

def run_scan():
    with _scan_lock:
        if scan_status["running"]: return
        scan_status["running"] = True
    try:
        scan_status["last_scan"] = datetime.utcnow().isoformat()
        cfg = _active_cfg
        pl = []
        if cfg["pairs"] in ("all","forex"):
            for p,s in FOREX_PAIRS.items():
                if not cfg["pair_name"] or cfg["pair_name"]==p: pl.append((p,s,"forex"))
        if cfg["pairs"] in ("all","crypto"):
            for p,s in CRYPTO_PAIRS.items():
                if not cfg["pair_name"] or cfg["pair_name"]==p: pl.append((p,s,"crypto"))
        tfs = {cfg["timeframe"]:TIMEFRAMES[cfg["timeframe"]]} if cfg["timeframe"]!="all" else TIMEFRAMES
        for pname,ysym,ptype in pl:
            scan_status["scanning_pair"] = pname
            for tf_key,tf_cfg in tfs.items():
                sig = _scan_pair(pname,ysym,tf_key,tf_cfg,ptype,cfg["risk"])
                if sig:
                    dup = [s for s in signal_store if s["pair"]==pname and s["tf_key"]==tf_key and s["status"]=="active"]
                    if not dup:
                        signal_store.appendleft(sig)
                        stats_store["total"]  +=1; stats_store["pending"]+=1
                        pending_checks.append(sig)
    finally:
        scan_status["running"]=False; scan_status["scanning_pair"]=None

def check_results():
    now = datetime.utcnow()
    done = []
    for sig in list(pending_checks):
        if now < datetime.fromisoformat(sig["resolve_at"]): continue
        ep = _get_price(sig["yf_symbol"])
        if ep is None: continue
        entry = sig.get("entry_raw")
        if entry:
            win = (sig["direction"]=="AL" and ep>entry) or (sig["direction"]=="SAT" and ep<entry)
            sig["status"]     = "win" if win else "loss"
            sig["exit_price"] = _fmt(sig["pair"],ep)
            sig["pnl"]        = round(ep-entry if sig["direction"]=="AL" else entry-ep, 6)
            if win: stats_store["wins"]+=1
            else:   stats_store["losses"]+=1
            stats_store["pending"] = max(0,stats_store["pending"]-1)
        done.append(sig)
    for s in done:
        if s in pending_checks: pending_checks.remove(s)

def _auto_loop():
    while True:
        time.sleep(15)
        try:
            if _active_cfg["active"]:
                run_scan()
                time.sleep(max(0,_active_cfg["interval"]-15))
            check_results()
        except Exception as e:
            app.logger.error(f"Loop hatası: {e}")

threading.Thread(target=_auto_loop, daemon=True).start()

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/register", methods=["POST"])
def register():
    d = request.get_json() or {}
    email,name,pw,lang = d.get("email","").strip().lower(),d.get("name","").strip(),d.get("password",""),d.get("lang","tr")
    if not email or not pw or not name: return jsonify({"ok":False,"error":"Tüm alanlar zorunlu"}),400
    with _lock:
        if email in _users_db: return jsonify({"ok":False,"error":"Bu e-posta zaten kayıtlı"}),400
        _users_db[email] = {"password":_hash(pw),"name":name,"lang":lang,"theme":"dark","created":datetime.utcnow().isoformat()}
    session.update({"user":email,"name":name,"lang":lang,"theme":"dark"})
    return jsonify({"ok":True,"name":name,"lang":lang,"theme":"dark"})

@app.route("/api/login", methods=["POST"])
def login():
    d = request.get_json() or {}
    email,pw = d.get("email","").strip().lower(),d.get("password","")
    with _lock: u = _users_db.get(email)
    if not u or u["password"]!=_hash(pw): return jsonify({"ok":False,"error":"E-posta veya şifre hatalı"}),401
    session.update({"user":email,"name":u["name"],"lang":u.get("lang","tr"),"theme":u.get("theme","dark")})
    return jsonify({"ok":True,"name":u["name"],"lang":u["lang"],"theme":u["theme"]})

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear(); return jsonify({"ok":True})

@app.route("/api/settings", methods=["POST"])
def update_settings():
    if not session.get("user"): return jsonify({"error":"Giriş gerekli"}),403
    d,email = request.get_json() or {},session["user"]
    with _lock:
        if email in _users_db:
            for k in ("lang","theme"):
                if k in d: _users_db[email][k]=d[k]; session[k]=d[k]
    return jsonify({"ok":True,"lang":session.get("lang"),"theme":session.get("theme")})

@app.route("/api/me")
def me():
    if not session.get("user"): return jsonify({"auth":False})
    return jsonify({"auth":True,"name":session.get("name"),"lang":session.get("lang","tr"),"theme":session.get("theme","dark")})

@app.route("/api/scan", methods=["POST"])
def start_scan():
    if not session.get("user"): return jsonify({"error":"Giriş gerekli"}),403
    d = request.get_json() or {}
    _active_cfg.update({"active":True,"pairs":d.get("pairs","all"),"timeframe":d.get("timeframe","all"),"risk":d.get("risk","high"),"pair_name":d.get("pair_name") or None})
    threading.Thread(target=run_scan,daemon=True).start()
    return jsonify({"ok":True})

@app.route("/api/stop", methods=["POST"])
def stop_scan():
    if not session.get("user"): return jsonify({"error":"Giriş gerekli"}),403
    _active_cfg["active"]=False; return jsonify({"ok":True})

@app.route("/api/signals")
def get_signals():
    if not session.get("user"): return jsonify({"error":"Giriş gerekli"}),403
    check_results()
    w,l = stats_store["wins"],stats_store["losses"]
    wr  = round(w/(w+l)*100,1) if (w+l)>0 else 0
    resolved = [s for s in signal_store if s["status"] in ("win","loss")]
    chart = [{"i":i+1,"win":s["status"]=="win","pair":s["pair"],"dir":s["direction"]} for i,s in enumerate(reversed(list(resolved)[:20]))]
    return jsonify({"signals":list(signal_store),"stats":{**stats_store,"win_rate":wr},"status":scan_status,"chart":chart})

@app.route("/api/clear", methods=["POST"])
def clear_signals():
    if not session.get("user"): return jsonify({"error":"Giriş gerekli"}),403
    signal_store.clear(); pending_checks.clear()
    stats_store.update({"total":0,"wins":0,"losses":0,"pending":0})
    return jsonify({"ok":True})

@app.route("/api/pairs")
def get_pairs():
    return jsonify({"forex":list(FOREX_PAIRS.keys()),"crypto":list(CRYPTO_PAIRS.keys()),"timeframes":list(TIMEFRAMES.keys())})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)), debug=False)
 
