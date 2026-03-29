"""
╔══════════════════════════════════════════════════════════════╗
║           APEX SIGNAL ENGINE v2.0                           ║
║  Katmanlı filtre sistemi ile maksimum win rate              ║
║  Binary Options için optimize edilmiş strateji             ║
╚══════════════════════════════════════════════════════════════╝

Strateji Mimarisi:
─────────────────
  Katman 1 │ Trend Yapısı   │ EMA hizalaması + gerçek ADX
  Katman 2 │ Momentum       │ RSI zone + MACD histogram ivmesi
  Katman 3 │ Osilatör Uyumu │ Stochastic + CCI birlikte bakılır
  Katman 4 │ Volatilite     │ Bollinger genişliği + ATR filtresi
  Katman 5 │ Fiyat Aksiyonu │ Son 3 mumun yönsel baskısı
  Katman 6 │ Hacim          │ Ortalama üstü hacim konfirmasyonu

  Minimum sinyal eşiği: 7 / 10 puan
  Orta risk:            5 / 10
  Yüksek risk:          4 / 10  (düşük filtre, daha fazla sinyal)
"""

from __future__ import annotations
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# YARDIMCI: güvenli son değer okuma
# ─────────────────────────────────────────────────────────────
def _last(series: pd.Series, default: float = 0.0) -> float:
    try:
        v = series.iloc[-1]
        return float(v) if pd.notna(v) else default
    except Exception:
        return default


def _prev(series: pd.Series, n: int = 1, default: float = 0.0) -> float:
    try:
        v = series.iloc[-(n + 1)]
        return float(v) if pd.notna(v) else default
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────
# KATMAN 1: EMA Trend Yapısı + Gerçek ADX
# ─────────────────────────────────────────────────────────────
def _layer_trend(df: pd.DataFrame) -> tuple[int, float, list[str]]:
    """
    Puan: 0-3
    EMA8 > EMA21 > EMA50 → güçlü yükseliş trendi (+2)
    EMA8 < EMA21 < EMA50 → güçlü düşüş trendi (+2)
    Sadece kısmi hizalanma → (+1)
    ADX > 25 → trend güçlü (+1 bonus)
    ADX > 40 → çok güçlü (+1 ekstra bonus, max 3 kalır)
    """
    details: list[str] = []
    score   = 0
    adx_val = 0.0
    close, high, low = df["close"], df["high"], df["low"]

    # EMA hesapla
    try:
        ema8  = close.ewm(span=8,  adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        e8  = _last(ema8)
        e21 = _last(ema21)
        e50 = _last(ema50)

        if e8 > e21 > e50:
            score += 2
            details.append(f"EMA8>EMA21>EMA50 ✅ (güçlü yükseliş)")
        elif e8 < e21 < e50:
            score += 2
            details.append(f"EMA8<EMA21<EMA50 ✅ (güçlü düşüş)")
        elif e8 > e21:
            score += 1
            details.append(f"EMA8>EMA21 (kısmi yükseliş)")
        elif e8 < e21:
            score += 1
            details.append(f"EMA8<EMA21 (kısmi düşüş)")
        else:
            details.append("EMA hizalaması nötr")
    except Exception as e:
        logger.debug(f"EMA hesap hatası: {e}")
        details.append("EMA hesaplanamadı")

    # Gerçek ADX (Wilder yöntemi)
    try:
        period = 14
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        dm_plus  = (high - high.shift(1)).clip(lower=0)
        dm_minus = (low.shift(1) - low).clip(lower=0)
        # İki yönde de DM varsa küçüğü sıfırla
        mask = dm_plus >= dm_minus
        dm_plus  = dm_plus.where(mask, 0)
        dm_minus = dm_minus.where(~mask, 0)

        atr14  = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        dip14  = dm_plus.ewm(alpha=1/period,  min_periods=period, adjust=False).mean()
        dim14  = dm_minus.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        di_plus  = 100 * dip14 / (atr14 + 1e-9)
        di_minus = 100 * dim14 / (atr14 + 1e-9)
        dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
        adx_ser  = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        adx_val  = float(_last(adx_ser))

        if adx_val > 40:
            score = min(score + 1, 3)
            details.append(f"ADX={adx_val:.1f} 🔥 (çok güçlü trend)")
        elif adx_val > 25:
            score = min(score + 1, 3)
            details.append(f"ADX={adx_val:.1f} ✅ (güçlü trend)")
        else:
            details.append(f"ADX={adx_val:.1f} (zayıf/yatay trend)")

        score = min(score, 3)
    except Exception as e:
        logger.debug(f"ADX hesap hatası: {e}")
        details.append("ADX hesaplanamadı")

    return score, adx_val, details


# ─────────────────────────────────────────────────────────────
# KATMAN 2: RSI Momentum + MACD Histogram İvmesi
# ─────────────────────────────────────────────────────────────
def _layer_momentum(df: pd.DataFrame) -> tuple[int, str, list[str]]:
    """
    Puan: 0-2
    Yön: "AL" | "SAT" | "NÖTR"

    RSI aşırı bölgeden döndü mü? + MACD histogramı büyüyor mu?
    Her ikisi de aynı yönde → +2
    Sadece biri uyumlu    → +1
    Çelişiyor             → 0
    """
    details: list[str] = []
    score   = 0
    rsi_dir = "NÖTR"
    mac_dir = "NÖTR"
    close   = df["close"]

    # RSI (14) — aşırı alım/satım → dönüş beklentisi
    try:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss + 1e-9)
        rsi   = 100 - 100 / (1 + rs)

        r_now  = _last(rsi)
        r_prev = _prev(rsi)

        # Sıkılaştırılmış eşikler (35/65 yerine 30/70)
        if r_prev <= 30 and r_now > 30:
            rsi_dir = "AL"
            details.append(f"RSI aşırı satımdan çıkış: {r_now:.1f} ✅")
        elif r_prev >= 70 and r_now < 70:
            rsi_dir = "SAT"
            details.append(f"RSI aşırı alımdan çıkış: {r_now:.1f} ✅")
        elif r_now < 35:
            rsi_dir = "AL"
            details.append(f"RSI={r_now:.1f} aşırı satım bölgesinde")
        elif r_now > 65:
            rsi_dir = "SAT"
            details.append(f"RSI={r_now:.1f} aşırı alım bölgesinde")
        else:
            details.append(f"RSI={r_now:.1f} nötr")
    except Exception as e:
        logger.debug(f"RSI hatası: {e}")

    # MACD histogram ivmesi (12,26,9)
    try:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram   = macd_line - signal_line

        h_now  = _last(histogram)
        h_prev = _prev(histogram)
        h_pp   = _prev(histogram, 2)

        # Histogram büyüme ivmesi (son 3 mum aynı yönde büyüyor mu?)
        if h_now > h_prev > h_pp and h_now > 0:
            mac_dir = "AL"
            details.append(f"MACD histogram büyüyor ↑ ✅")
        elif h_now < h_prev < h_pp and h_now < 0:
            mac_dir = "SAT"
            details.append(f"MACD histogram küçülüyor ↓ ✅")
        elif h_now > 0 and h_prev < 0:
            mac_dir = "AL"
            details.append(f"MACD sıfır çizgisi yukarı geçti ✅")
        elif h_now < 0 and h_prev > 0:
            mac_dir = "SAT"
            details.append(f"MACD sıfır çizgisi aşağı geçti ✅")
        else:
            details.append("MACD nötr")
    except Exception as e:
        logger.debug(f"MACD hatası: {e}")

    # Yön uyumu
    if rsi_dir == mac_dir and rsi_dir != "NÖTR":
        score   = 2
        overall = rsi_dir
    elif rsi_dir != "NÖTR":
        score   = 1
        overall = rsi_dir
    elif mac_dir != "NÖTR":
        score   = 1
        overall = mac_dir
    else:
        overall = "NÖTR"

    return score, overall, details


# ─────────────────────────────────────────────────────────────
# KATMAN 3: Stochastic + CCI Osilatör Uyumu
# ─────────────────────────────────────────────────────────────
def _layer_oscillators(df: pd.DataFrame) -> tuple[int, str, list[str]]:
    """
    Puan: 0-2
    Stoch K/D çaprazlaması + CCI aşırı bölgeden dönüş
    """
    details: list[str] = []
    score   = 0
    sto_dir = "NÖTR"
    cci_dir = "NÖTR"
    close, high, low = df["close"], df["high"], df["low"]

    # Stochastic (14,3,3) — K/D kesişimi
    try:
        low14  = low.rolling(14).min()
        high14 = high.rolling(14).max()
        k_raw  = (close - low14) / (high14 - low14 + 1e-9) * 100
        k_line = k_raw.rolling(3).mean()
        d_line = k_line.rolling(3).mean()

        k_now  = _last(k_line)
        d_now  = _last(d_line)
        k_prev = _prev(k_line)
        d_prev = _prev(d_line)

        # K çizgisi D'yi yukarı kesiyor ve aşırı satım bölgesinde
        if k_prev <= d_prev and k_now > d_now and k_now < 40:
            sto_dir = "AL"
            details.append(f"Stoch K/D yukarı kesişim (K={k_now:.1f}) ✅")
        # K çizgisi D'yi aşağı kesiyor ve aşırı alım bölgesinde
        elif k_prev >= d_prev and k_now < d_now and k_now > 60:
            sto_dir = "SAT"
            details.append(f"Stoch K/D aşağı kesişim (K={k_now:.1f}) ✅")
        elif k_now < 25:
            sto_dir = "AL"
            details.append(f"Stoch K={k_now:.1f} aşırı satım")
        elif k_now > 75:
            sto_dir = "SAT"
            details.append(f"Stoch K={k_now:.1f} aşırı alım")
        else:
            details.append(f"Stoch K={k_now:.1f} nötr")
    except Exception as e:
        logger.debug(f"Stoch hatası: {e}")

    # CCI (20) — sıfır çizgisi geçişi
    try:
        tp  = (high + low + close) / 3
        sma = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        cci = (tp - sma) / (0.015 * mad + 1e-9)

        c_now  = _last(cci)
        c_prev = _prev(cci)

        if c_prev < -100 and c_now > -100:
            cci_dir = "AL"
            details.append(f"CCI aşırı satımdan çıkış ({c_now:.0f}) ✅")
        elif c_prev > 100 and c_now < 100:
            cci_dir = "SAT"
            details.append(f"CCI aşırı alımdan çıkış ({c_now:.0f}) ✅")
        elif c_now < -80:
            cci_dir = "AL"
            details.append(f"CCI={c_now:.0f} aşırı satım bölgesi")
        elif c_now > 80:
            cci_dir = "SAT"
            details.append(f"CCI={c_now:.0f} aşırı alım bölgesi")
        else:
            details.append(f"CCI={c_now:.0f} nötr")
    except Exception as e:
        logger.debug(f"CCI hatası: {e}")

    # Uyum kontrolü
    if sto_dir == cci_dir and sto_dir != "NÖTR":
        score   = 2
        overall = sto_dir
    elif sto_dir != "NÖTR":
        score   = 1
        overall = sto_dir
    elif cci_dir != "NÖTR":
        score   = 1
        overall = cci_dir
    else:
        overall = "NÖTR"

    return score, overall, details


# ─────────────────────────────────────────────────────────────
# KATMAN 4: Bollinger Bands + ATR Volatilite Filtresi
# ─────────────────────────────────────────────────────────────
def _layer_volatility(df: pd.DataFrame) -> tuple[int, str, list[str]]:
    """
    Puan: 0-2
    BB sıkışma sonrası kırılım + ATR momentum filtresi

    Sıkışma (BB dar) + yönlü kırılım = güçlü sinyal
    Normal BB + band teması = orta sinyal
    BB çok geniş (yüksek vol) = sinyal kalitesizdir
    """
    details: list[str] = []
    score   = 0
    vol_dir = "NÖTR"
    close, high, low = df["close"], df["high"], df["low"]

    try:
        mid   = close.rolling(20).mean()
        std   = close.rolling(20).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        bw    = (upper - lower) / (mid + 1e-9)  # Band genişliği (normalize)

        price  = _last(close)
        u_now  = _last(upper)
        l_now  = _last(lower)
        m_now  = _last(mid)
        bw_now = _last(bw)
        bw_avg = float(bw.rolling(50).mean().iloc[-1]) if len(bw) >= 50 else bw_now

        # ATR (14) — son hareketlilik
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_now = _last(atr)
        atr_avg = float(atr.rolling(50).mean().iloc[-1]) if len(atr) >= 50 else atr_now

        # BB sıkışma tespiti (son bant genişliği ortalamanın %70'inden azsa)
        is_squeeze = bw_now < bw_avg * 0.70

        if is_squeeze:
            details.append("BB sıkışma var 🔒")
            # Sıkışma + kırılım = çok güçlü
            if price >= u_now:
                vol_dir = "SAT"  # Üst banda dokunuş → sat sinyali (mean reversion)
                score = 2
                details.append(f"BB sıkışma sonrası üst band kırıldı ✅")
            elif price <= l_now:
                vol_dir = "AL"
                score = 2
                details.append(f"BB sıkışma sonrası alt band kırıldı ✅")
        else:
            # Normal BB analizi
            if price <= l_now and atr_now > atr_avg * 0.8:
                vol_dir = "AL"
                score = 1
                details.append(f"Fiyat BB alt bandında (bw={bw_now:.3f}) ✅")
            elif price >= u_now and atr_now > atr_avg * 0.8:
                vol_dir = "SAT"
                score = 1
                details.append(f"Fiyat BB üst bandında (bw={bw_now:.3f}) ✅")
            elif bw_now > bw_avg * 2.0:
                # Çok yüksek volatilite → sinyal güvenilmez
                details.append(f"BB çok geniş (vol çok yüksek, sinyal zayıf)")
                score = -1  # Ceza puanı
            else:
                details.append(f"BB içinde (bw={bw_now:.3f}) nötr")

    except Exception as e:
        logger.debug(f"BB hatası: {e}")

    return score, vol_dir, details


# ─────────────────────────────────────────────────────────────
# KATMAN 5: Fiyat Aksiyonu (Son 3 Mumun Analizi)
# ─────────────────────────────────────────────────────────────
def _layer_price_action(df: pd.DataFrame) -> tuple[int, str, list[str]]:
    """
    Puan: 0-2
    Son 3 mumun yönsel baskısını ve formasyonlarını analiz eder.

    Aranılan formasyonlar:
    - Engulfing (yutan mum)
    - Pin bar (uzun fitil, küçük gövde)
    - 3 ardışık aynı yönlü mum (momentum)
    - Inside bar kırılımı
    """
    details: list[str] = []
    score   = 0
    pa_dir  = "NÖTR"

    try:
        o = df["open"]
        h = df["high"]
        l = df["low"]
        c = df["close"]

        # Son 4 mumun verileri
        o1, o2, o3 = _last(o, 0), _prev(o, 1), _prev(o, 2)
        c1, c2, c3 = _last(c, 0), _prev(c, 1), _prev(c, 2)
        h1, h2     = _last(h, 0), _prev(h, 1)
        l1, l2     = _last(l, 0), _prev(l, 1)

        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        range1 = h1 - l1 + 1e-9
        range2 = h2 - l2 + 1e-9
        upper_wick1 = h1 - max(c1, o1)
        lower_wick1 = min(c1, o1) - l1

        # 1. Bullish Engulfing (son mum öncekini tamamen yutuyor)
        if c1 > o1 and c2 < o2 and c1 > o2 and o1 < c2:
            pa_dir = "AL"
            score  = 2
            details.append("Bullish Engulfing formasyon ✅")

        # 2. Bearish Engulfing
        elif c1 < o1 and c2 > o2 and c1 < o2 and o1 > c2:
            pa_dir = "SAT"
            score  = 2
            details.append("Bearish Engulfing formasyon ✅")

        # 3. Bullish Pin Bar (uzun alt fitil, küçük gövde)
        elif lower_wick1 > body1 * 2 and lower_wick1 > upper_wick1 * 2:
            pa_dir = "AL"
            score  = 2
            details.append("Bullish Pin Bar ✅")

        # 4. Bearish Pin Bar (uzun üst fitil, küçük gövde)
        elif upper_wick1 > body1 * 2 and upper_wick1 > lower_wick1 * 2:
            pa_dir = "SAT"
            score  = 2
            details.append("Bearish Pin Bar ✅")

        # 5. 3 ardışık yükselen mum (momentum)
        elif c1 > o1 and c2 > o2 and c3 > o3:
            pa_dir = "AL"
            score  = 1
            details.append("3 ardışık yükselen mum (momentum) ✅")

        # 6. 3 ardışık düşen mum
        elif c1 < o1 and c2 < o2 and c3 < o3:
            pa_dir = "SAT"
            score  = 1
            details.append("3 ardışık düşen mum (momentum) ✅")

        # 7. Son mum güçlü (gövde > range'in %60'ı)
        elif body1 / range1 > 0.60:
            if c1 > o1:
                pa_dir = "AL"
                score  = 1
                details.append(f"Güçlü boğa mumu ({body1/range1*100:.0f}% gövde) ✅")
            else:
                pa_dir = "SAT"
                score  = 1
                details.append(f"Güçlü ayı mumu ({body1/range1*100:.0f}% gövde) ✅")
        else:
            details.append("Belirgin fiyat aksiyonu yok")

    except Exception as e:
        logger.debug(f"Fiyat aksiyonu hatası: {e}")

    return score, pa_dir, details


# ─────────────────────────────────────────────────────────────
# KATMAN 6: Hacim Konfirmasyonu
# ─────────────────────────────────────────────────────────────
def _layer_volume(df: pd.DataFrame) -> tuple[int, list[str]]:
    """
    Puan: 0-1
    Son mum hacmi ortalamanın üstündeyse +1 (sinyal güvenilir)
    """
    details: list[str] = []
    score   = 0

    try:
        if "volume" not in df.columns:
            return 0, ["Hacim verisi yok"]

        vol     = df["volume"]
        v_now   = _last(vol)
        v_avg   = float(vol.rolling(20).mean().iloc[-1])

        if v_now > v_avg * 1.5:
            score = 1
            details.append(f"Hacim ortalamanın {v_now/v_avg:.1f}x üstünde 🔥 ✅")
        elif v_now > v_avg * 1.0:
            score = 1
            details.append(f"Hacim ortalama üstünde ({v_now/v_avg:.1f}x) ✅")
        else:
            details.append(f"Hacim ortalama altında ({v_now/v_avg:.1f}x)")
    except Exception as e:
        logger.debug(f"Hacim hatası: {e}")

    return score, details


# ─────────────────────────────────────────────────────────────
# ANA ANALİZ FONKSİYONU (Tüm katmanları birleştirir)
# ─────────────────────────────────────────────────────────────
def analyze_apex(df: pd.DataFrame) -> dict:
    """
    APEX Signal Engine ana analiz fonksiyonu.

    Returns:
        {
            "direction": "AL" | "SAT" | None,
            "score": int,          # Toplam ham puan
            "max_score": int,      # Maksimum mümkün puan
            "confidence": float,   # 0.0 - 1.0 arası güven skoru
            "adx": float,          # Trend gücü
            "details": [str],      # Tüm açıklamalar
            "layer_scores": dict,  # Katman bazlı puanlar
            "signal_valid": bool,  # Sinyal yayınlanabilir mi?
        }
    """
    if df is None or len(df) < 55:
        return _null_result("Yetersiz veri (min 55 mum gerekli)")

    details:     list[str] = []
    layer_scores: dict     = {}

    # ── Katman 1: Trend ─────────────────────────────────────
    t_score, adx_val, t_details = _layer_trend(df)
    details.extend(t_details)
    layer_scores["trend"] = t_score

    # Zayıf trend → tüm diğer sinyaller güvenilmez (binary'de trend kritik)
    # ADX < 15 ise sinyal üretme
    if adx_val < 15:
        return _null_result(f"ADX={adx_val:.1f} çok zayıf, sinyal yok")

    # ── Katman 2: Momentum ──────────────────────────────────
    m_score, m_dir, m_details = _layer_momentum(df)
    details.extend(m_details)
    layer_scores["momentum"] = m_score

    # ── Katman 3: Osilatörler ───────────────────────────────
    o_score, o_dir, o_details = _layer_oscillators(df)
    details.extend(o_details)
    layer_scores["oscillators"] = o_score

    # ── Katman 4: Volatilite ────────────────────────────────
    v_score, v_dir, v_details = _layer_volatility(df)
    details.extend(v_details)
    layer_scores["volatility"] = v_score

    # ── Katman 5: Fiyat Aksiyonu ─────────────────────────────
    p_score, p_dir, p_details = _layer_price_action(df)
    details.extend(p_details)
    layer_scores["price_action"] = p_score

    # ── Katman 6: Hacim ─────────────────────────────────────
    h_score, h_details = _layer_volume(df)
    details.extend(h_details)
    layer_scores["volume"] = h_score

    # ── Yön Oylama Sistemi ───────────────────────────────────
    al_votes  = 0
    sat_votes = 0
    al_weight = 0.0
    sat_weight = 0.0

    # Ağırlıklı oylama (trend katmanı 2x ağırlıklı)
    dirs_weights = [
        (m_dir, m_score, 1.0),   # Momentum
        (o_dir, o_score, 1.0),   # Osilatör
        (v_dir, v_score, 0.8),   # Volatilite
        (p_dir, p_score, 1.2),   # Fiyat aksiyonu (en önemli)
    ]
    for d, s, w in dirs_weights:
        if d == "AL":
            al_votes  += 1
            al_weight += s * w
        elif d == "SAT":
            sat_votes  += 1
            sat_weight += s * w

    # Trend yönünü EMA yapısından al
    try:
        ema8  = df["close"].ewm(span=8,  adjust=False).mean()
        ema21 = df["close"].ewm(span=21, adjust=False).mean()
        e8, e21 = _last(ema8), _last(ema21)
        if e8 > e21:
            trend_dir = "AL"
        elif e8 < e21:
            trend_dir = "SAT"
        else:
            trend_dir = "NÖTR"
    except Exception:
        trend_dir = "NÖTR"

    # Ana yön kararı
    if al_weight > sat_weight:
        direction = "AL"
    elif sat_weight > al_weight:
        direction = "SAT"
    else:
        direction = "NÖTR"

    # Trend ile çelişiyor mu? (Kontr-trend = pullback senaryosu)
    trend_conflict = (trend_dir != "NÖTR" and direction != "NÖTR"
                      and trend_dir != direction)
    if trend_conflict:
        # Kontr-trend'de skor cezası ver ama tamamen iptal etme
        # (binary'de pullback sinyalleri de değerlidir, ancak daha yüksek eşik ister)
        raw_score_before_penalty = (
            t_score + m_score + o_score + max(v_score, 0) + p_score + h_score
            + (v_score if v_score < 0 else 0)
        )
        if raw_score_before_penalty < 7:
            # Zayıf kontr-trend → iptal
            details.append("⚠️ Kontr-trend & düşük skor → iptal")
            return _null_result("Kontr-trend sinyal (zayıf) — yayınlanmadı")
        else:
            # Güçlü kontr-trend (pullback) → skor cezası ile devam
            details.append("⚠️ Kontr-trend (pullback) — skor düşürüldü")

    # ── Toplam Skor ──────────────────────────────────────────
    raw_score = (
        t_score            # max 3
        + m_score          # max 2
        + o_score          # max 2
        + max(v_score, 0)  # max 2 (negatif olabilir)
        + p_score          # max 2
        + h_score          # max 1
    )
    # BB yüksek volatilite cezasını uygula
    if v_score < 0:
        raw_score += v_score  # zaten negatif

    # Kontr-trend cezası (pullback sinyali = -2 puan)
    if trend_conflict:
        raw_score = max(0, raw_score - 2)

    max_score   = 12
    confidence  = max(0.0, min(1.0, raw_score / max_score))

    # ── Sinyal Geçerlilik Eşiği ──────────────────────────────
    # Risk seviyesine göre dinamik eşik main.py'de uygulanacak
    signal_valid = (
        direction != "NÖTR"
        and raw_score >= 4          # Mutlak minimum
        and t_score >= 1            # Trend katmanından en az 1 puan
        and (m_score + o_score) >= 2  # Momentum + Osilatör birlikte en az 2
    )

    return {
        "direction":    direction if signal_valid else None,
        "score":        raw_score,
        "max_score":    max_score,
        "confidence":   confidence,
        "adx":          adx_val,
        "details":      details,
        "layer_scores": layer_scores,
        "signal_valid": signal_valid,
        "al_weight":    al_weight,
        "sat_weight":   sat_weight,
    }


def _null_result(reason: str) -> dict:
    return {
        "direction":    None,
        "score":        0,
        "max_score":    12,
        "confidence":   0.0,
        "adx":          0.0,
        "details":      [reason],
        "layer_scores": {},
        "signal_valid": False,
        "al_weight":    0.0,
        "sat_weight":   0.0,
    }


# ─────────────────────────────────────────────────────────────
# Risk seviyesi → minimum skor eşiği
# ─────────────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "low":  8,   # Düşük risk: çok seçici, yüksek güven ister
    "mid":  6,   # Orta risk: dengeli
    "high": 4,   # Yüksek risk: daha fazla sinyal, daha az filtre
}

def score_to_risk_level(score: int) -> str:
    """Verilen skora göre risk seviyesini belirle."""
    if score >= 8:
        return "low"    # Güçlü sinyal → düşük risk
    if score >= 6:
        return "mid"
    return "high"
