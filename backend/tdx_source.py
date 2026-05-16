"""
通达信行情（pytdx）拉取日线、1 分钟 K 线；失败时由 scraper 回退其它源。
"""
from __future__ import annotations

import datetime as dt
from typing import List, Optional, Tuple

# 常用通达信行情前置（可多 host 轮询）
_TDX_HOSTS: Tuple[Tuple[str, int], ...] = (
    ("119.147.171.206", 7709),
    ("115.159.198.242", 7709),
    ("124.70.75.113", 7709),
    ("106.120.74.86", 7711),
)


def _pure_code_and_market(code: str) -> Tuple[int, str]:
    c = code.replace(".SH", "").replace(".SZ", "").strip()
    market = 1 if c.startswith("6") else 0
    return market, c


def _try_connect_api():
    try:
        from pytdx.hq import TdxHq_API
    except ImportError:
        return None
    api = TdxHq_API()
    for host, port in _TDX_HOSTS:
        try:
            if api.connect(host, port):
                return api
        except Exception:
            try:
                api.disconnect()
            except Exception:
                pass
    return None


def _bars_to_klines_day(bars: List[dict]) -> List[dict]:
    out = []
    for b in bars or []:
        dt_s = b.get("datetime") or ""
        if isinstance(dt_s, bytes):
            dt_s = dt_s.decode("utf-8", errors="ignore")
        dt_s = str(dt_s).strip()
        if len(dt_s) >= 10:
            date_part = dt_s[:10]
        else:
            continue
        try:
            vol = int(float(b.get("vol", b.get("volume", 0)) or 0))
        except (TypeError, ValueError):
            vol = 0
        try:
            amt = float(b.get("amount", 0) or 0)
        except (TypeError, ValueError):
            amt = 0.0
        out.append({
            "date": date_part,
            "open": float(b["open"]),
            "close": float(b["close"]),
            "high": float(b["high"]),
            "low": float(b["low"]),
            "volume": vol,
            "amount": amt,
        })
    out.sort(key=lambda x: x["date"])
    return out


def _bars_to_klines_minute(bars: List[dict]) -> List[dict]:
    """1 分钟 OHLC，date 为 YYYY-MM-DD HH:MM:SS（与 Mongo 分时兼容）。"""
    out = []
    for b in bars or []:
        dt_s = b.get("datetime") or ""
        if isinstance(dt_s, bytes):
            dt_s = dt_s.decode("utf-8", errors="ignore")
        dt_s = str(dt_s).strip()
        if len(dt_s) < 10:
            continue
        try:
            vol = int(float(b.get("vol", b.get("volume", 0)) or 0))
        except (TypeError, ValueError):
            vol = 0
        try:
            amt = float(b.get("amount", 0) or 0)
        except (TypeError, ValueError):
            amt = 0.0
        out.append({
            "date": dt_s if len(dt_s) > 10 else dt_s + " 00:00:00",
            "open": float(b["open"]),
            "close": float(b["close"]),
            "high": float(b["high"]),
            "low": float(b["low"]),
            "volume": vol,
            "amount": amt,
        })
    out.sort(key=lambda x: x["date"])
    return out


def fetch_daily_kline(code: str, count: int = 800) -> Optional[List[dict]]:
    """日 K；pytdx category 4 = 日线。"""
    api = _try_connect_api()
    if not api:
        return None
    market, pure = _pure_code_and_market(code)
    try:
        chunks: List[dict] = []
        remain = min(int(count), 800)
        pos = 0
        while remain > 0:
            n = min(800, remain)
            part = api.get_security_bars(4, market, pure, pos, n)
            if not part:
                break
            chunks.extend(part)
            pos += len(part)
            remain -= len(part)
            if len(part) < n:
                break
        return _bars_to_klines_day(chunks) if chunks else None
    except Exception as e:
        print(f"[tdx] daily failed {code}: {e}")
        return None
    finally:
        try:
            api.disconnect()
        except Exception:
            pass


def fetch_minute_1m(code: str, count: int = 800) -> Optional[List[dict]]:
    """最近 1 分钟 K 线（含多交易日片段）；pytdx category 8 = 1 分钟。"""
    api = _try_connect_api()
    if not api:
        return None
    market, pure = _pure_code_and_market(code)
    try:
        chunks: List[dict] = []
        remain = min(int(count), 800)
        pos = 0
        while remain > 0:
            n = min(800, remain)
            part = api.get_security_bars(8, market, pure, pos, n)
            if not part:
                break
            chunks.extend(part)
            pos += len(part)
            remain -= len(part)
            if len(part) < n:
                break
        return _bars_to_klines_minute(chunks) if chunks else None
    except Exception as e:
        print(f"[tdx] minute failed {code}: {e}")
        return None
    finally:
        try:
            api.disconnect()
        except Exception:
            pass


def fetch_minute_1m_paged(
    code: str,
    stop_calendar_before: Optional[str] = None,
    max_chunks: int = 120,
) -> Optional[List[dict]]:
    """
    从最新一根起向后翻页拉 1 分钟 K，合并去重；可选 stop_calendar_before=YYYY-MM-DD：
    当本批最老一根的日历日 **早于** 该日时停止（用于拉取含某日在内的历史窗口）。
    """
    api = _try_connect_api()
    if not api:
        return None
    market, pure = _pure_code_and_market(code)
    stop_d: Optional[dt.date] = None
    if stop_calendar_before:
        try:
            stop_d = dt.datetime.strptime(stop_calendar_before.strip()[:10], "%Y-%m-%d").date()
        except ValueError:
            stop_d = None
    try:
        by_dt: dict = {}
        pos = 0
        for _ in range(max_chunks):
            part = api.get_security_bars(8, market, pure, pos, 800)
            if not part:
                break
            rows = _bars_to_klines_minute(part)
            for r in rows:
                by_dt[str(r["date"]).strip()] = r
            pos += len(part)
            if stop_d is not None:
                oldest = None
                for r in rows:
                    ds = str(r.get("date", "")).strip()[:10]
                    try:
                        d0 = dt.datetime.strptime(ds, "%Y-%m-%d").date()
                    except ValueError:
                        continue
                    if oldest is None or d0 < oldest:
                        oldest = d0
                if oldest is not None and oldest < stop_d:
                    break
            if len(part) < 800:
                break
        if not by_dt:
            return None
        out = sorted(by_dt.values(), key=lambda x: str(x["date"]))
        return out
    except Exception as e:
        print(f"[tdx] minute paged failed {code}: {e}")
        return None
    finally:
        try:
            api.disconnect()
        except Exception:
            pass


def minute_bars_to_trend_shape(minute_klines: List[dict], pre_close: float = 0.0) -> List[dict]:
    """
    将 1 分钟 OHLC 转为与东方财富 trends 类似的「当日累积」分时结构（单交易日内）。
    仅用于当日分时展示兼容；按日期分组后取最后一组。
    """
    if not minute_klines:
        return []
    day_open = None
    day_high = -1e18
    day_low = 1e18
    out: List[dict] = []
    for k in minute_klines:
        o, c, h, l = float(k["open"]), float(k["close"]), float(k["high"]), float(k["low"])
        if day_open is None:
            day_open = o
        day_high = max(day_high, h)
        day_low = min(day_low, l)
        out.append({
            "date": k["date"],
            "open": day_open,
            "close": c,
            "high": day_high,
            "low": day_low,
            "volume": int(k.get("volume", 0)),
            "amount": float(k.get("amount", 0)),
            "pre_close": pre_close,
        })
    return out


def fetch_minute_today_trends(code: str) -> Optional[List[dict]]:
    """当日分时：取今日 1 分钟并转为 trends 形状；无前一日收则 pre_close=首日 open。"""
    bars = fetch_minute_1m(code, count=800)
    if not bars:
        return None
    today = dt.date.today().strftime("%Y-%m-%d")
    day_bars = [b for b in bars if str(b["date"]).startswith(today)]
    if not day_bars:
        day_bars = bars
    pre_close = 0.0
    first_date = str(day_bars[0]["date"]).split()[0]
    for b in bars:
        ds = str(b["date"]).split()[0]
        if ds < first_date:
            pre_close = float(b["close"])
    if pre_close <= 0 and day_bars:
        pre_close = float(day_bars[0]["open"])
    return minute_bars_to_trend_shape(day_bars, pre_close=pre_close)


def fetch_historical_minute_for_date(code: str, target_date: str) -> Optional[List[dict]]:
    """历史某日分时：分页 1 分钟直到覆盖 target_date，过滤后转 trends 形状（与东财分时兼容）。"""
    td = target_date.strip()[:10]
    bars = fetch_minute_1m_paged(code, stop_calendar_before=td, max_chunks=120)
    if not bars:
        return None
    day_bars = [b for b in bars if str(b["date"]).startswith(td)]
    if not day_bars:
        return None
    pre_close = 0.0
    for b in bars:
        ds = str(b["date"]).split()[0]
        if ds < td:
            pre_close = float(b["close"])
    if pre_close <= 0:
        pre_close = float(day_bars[0]["open"])
    return minute_bars_to_trend_shape(day_bars, pre_close=pre_close)


def tdx_available() -> bool:
    try:
        from pytdx.hq import TdxHq_API  # noqa: F401
        return True
    except ImportError:
        return False
