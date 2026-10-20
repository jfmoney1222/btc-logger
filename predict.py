import os, json, time
import numpy as np, pandas as pd, requests, joblib

CB = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2/markets"
OUT = "predictions.csv"
HEADERS = {"User-Agent": "btc-paper-logger"}


def fetch_candles(gran, n):
    end = pd.Timestamp.now(tz="UTC").floor(f"{gran}s")
    cur = end - pd.Timedelta(seconds=gran * n)
    rows = []
    while cur < end:
        nxt = min(cur + pd.Timedelta(seconds=gran * 300), end)
        params = {"granularity": gran, "start": cur.isoformat(), "end": nxt.isoformat()}
        for attempt in range(4):
            r = requests.get(CB, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                rows += r.json()
                break
            time.sleep(2)
        else:
            raise RuntimeError(f"Coinbase request failed: {r.status_code}")
        cur = nxt
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.drop_duplicates("time").sort_values("time").set_index("time")


def build_features(d15, m1, T):
    """Clues for the window starting at T, using only data from before T."""
    d = d15[d15.index < T]
    d = d.reindex(pd.date_range(d.index.min(), T - pd.Timedelta(minutes=15), freq="15min"))
    d["close"] = d["close"].ffill()
    for col in ["open", "high", "low"]:
        d[col] = d[col].fillna(d["close"])
    d["volume"] = d["volume"].fillna(0)
    c, o, h, l, v = d["close"], d["open"], d["high"], d["low"], d["volume"]

    ret = np.log(c).diff()
    base = pd.DataFrame(index=d.index)
    for k in range(0, 8):
        base[f"ret_{k}"] = ret.shift(k)
    for n in [4, 8, 16, 32, 96]:
        base[f"mom_{n}"] = np.log(c).diff(n)
    for n in [8, 32, 96]:
        base[f"vol_{n}"] = ret.rolling(n).std()
    base["volume_ratio"] = v / (v.rolling(96).mean() + 1e-9)
    base["range"] = (h - l) / c
    base["body"] = (c - o) / o
    base["close_pos"] = (c - l) / (h - l + 1e-9)
    base["up_frac_8"] = (c >= o).astype(float).rolling(8).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    base["rsi14"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    base = base.reindex(base.index.append(pd.DatetimeIndex([T]))).shift(1)
    F = base.loc[[T]].copy()
    hr = T.hour + T.minute / 60
    F["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    F["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    F["dow"] = T.dayofweek

    mm = m1[m1.index < T].tail(60)
    mm = mm.reindex(pd.date_range(mm.index.min(), T - pd.Timedelta(minutes=1), freq="1min"))
    mm["close"] = mm["close"].ffill()
    for col in ["open", "high", "low"]:
        mm[col] = mm[col].fillna(mm["close"])
    avg = (mm["open"] + mm["high"] + mm["low"] + mm["close"]) / 4
    lc = np.log(mm["close"])
    F["gap"] = lc.iloc[-1] - np.log(avg.iloc[-1])
    F["r1"] = lc.iloc[-1] - lc.iloc[-2]
    F["r3"] = lc.iloc[-1] - lc.iloc[-4]
    F["r5"] = lc.iloc[-1] - lc.iloc[-6]
    return F


def kalshi_market(T):
    r = requests.get(KALSHI, params={"series_ticker": "KXBTC15M", "status": "open", "limit": 50},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    target = T + pd.Timedelta(minutes=15)
    best, best_d = None, None
    for m in r.json().get("markets", []):
        ct = pd.Timestamp(m["close_time"])
        if ct.tzinfo is None:
            ct = ct.tz_localize("UTC")
        d = abs((ct - target).total_seconds())
        if best_d is None or d < best_d:
            best, best_d = m, d
    if best is None or best_d > 120:
        return "", "", ""
    keep = {k: v for k, v in best.items()
            if k.startswith(("yes_", "no_")) or k in
            ("last_price", "last_price_dollars", "volume", "volume_fp", "open_interest", "status")}
    return best.get("ticker", ""), best.get("close_time", ""), json.dumps(keep)


def main():
    bundle = joblib.load("model.joblib")
    now = pd.Timestamp.now(tz="UTC")
    T = now.floor("15min")

    if os.path.exists(OUT):
        done = pd.read_csv(OUT)
        if str(T) in set(done["window_start"].astype(str)):
            print("Window already logged, skipping:", T)
            return

    d15 = fetch_candles(900, 450)
    m1 = fetch_candles(60, 60)
    if d15.index.max() < T - pd.Timedelta(minutes=45) or m1.index.max() < T - pd.Timedelta(minutes=10):
        raise RuntimeError("Price data looks stale, not logging")

    F = build_features(d15, m1, T)
    X = F[bundle["cols"]]
    p_lr = float(bundle["lr"].predict_proba(X)[0, 1])
    p_gbm = float(bundle["gbm"].predict_proba(X)[0, 1])

    try:
        ticker, close_time, kjson = kalshi_market(T)
    except Exception as e:
        print("Kalshi lookup failed:", e)
        ticker, close_time, kjson = "", "", ""

    row = {"window_start": str(T), "run_time": str(now),
           "delay_min": round((now - T).total_seconds() / 60, 2),
           "p_gbm": round(p_gbm, 4), "p_lr": round(p_lr, 4),
           "kalshi_ticker": ticker, "kalshi_close_time": close_time, "kalshi_json": kjson}
    pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    print(row)


if __name__ == "__main__":
    main()
