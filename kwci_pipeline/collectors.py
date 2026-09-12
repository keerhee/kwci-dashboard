from __future__ import annotations

import base64
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from . import config

_COMTRADE_UA = "KWCI-pipeline/1.0 (UN Comtrade public preview)"


def ensure_dirs() -> None:
    for path in (config.DATA_DIR, config.RAW_DIR, config.OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def now_kst_label() -> str:
    return datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")


def load_survey_baseline() -> pd.DataFrame:
    rows = []
    for country, values in config.SAMPLE_SURVEY.items():
        for genre in config.GENRE_WEIGHTS:
            rows.append(
                {
                    "country": country,
                    "genre": genre,
                    "survey_score": values[genre],
                    "kf_score": values["kf"],
                    "source": "sample_baseline",
                }
            )
    return pd.DataFrame(rows)


def _text_matches_genre(text: str, genre: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in config.GENRE_KEYWORDS[genre])


_yt_idx = 0  # 현재 사용 중인 YouTube 키 인덱스


def _yt_get(url: str, params: dict):
    """YouTube GET — 한도 초과(403 quotaExceeded) 시 다음 키로 자동 회전."""
    global _yt_idx
    keys = config.YOUTUBE_API_KEYS or ([config.YOUTUBE_API_KEY] if config.YOUTUBE_API_KEY else [])
    if not keys:
        return requests.get(url, params=params, timeout=30)
    last = None
    for _ in range(len(keys)):
        last = requests.get(url, params={**params, "key": keys[_yt_idx]}, timeout=30)
        if last.status_code == 403:
            reason = ""
            try:
                reason = last.json()["error"]["errors"][0].get("reason", "")
            except Exception:  # noqa: BLE001
                pass
            if reason in ("quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"):
                _yt_idx = (_yt_idx + 1) % len(keys)
                continue
        return last
    return last


# ── YouTube 쿼터 예산 ──────────────────────────────────────────
# Data API v3 는 하루 10,000 units 다. videos.list 는 1 unit 이지만
# search.list 는 **100 units** 라, 15개국 × 8도메인 조합마다 검색으로 빠지면
# 12,000 units 가 되어 한도를 넘는다. 실제로 2026-06-28 수집분은 84건,
# 07-02 는 69건이 search_http_429 로 떨어졌다(120행 중 유효 28·43행).
#
# 그래서 둘을 더한다.
#   (1) search 호출에 **예산 상한**. 넘으면 결측으로 남기고 다음 실행에서 받는다
#   (2) search 결과를 디스크에 캐시. 인기 영상 구성은 하루 단위로 크게 안 변한다
YT_SEARCH_BUDGET = int(os.getenv("YOUTUBE_SEARCH_BUDGET", "60"))   # 60 × 100 = 6,000 units
YT_CACHE_DAYS = int(os.getenv("YOUTUBE_CACHE_DAYS", "7"))
_YT_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "cache" / "youtube_search.json"
_yt_state = {"search_calls": 0, "skipped": 0}


def _yt_cache_load() -> dict:
    try:
        return json.loads(_YT_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _yt_cache_save(cache: dict) -> None:
    try:
        _YT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _YT_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def collect_youtube_metrics(sample: bool = False) -> pd.DataFrame:
    if sample or not config.YOUTUBE_API_KEY:
        return _sample_youtube_metrics()

    rows = []
    cache = _yt_cache_load()
    _yt_today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
    _yt_state.update(search_calls=0, skipped=0)
    videos_endpoint = "https://www.googleapis.com/youtube/v3/videos"
    # 인기차트(mostPopular)를 pageToken으로 최대 YOUTUBE_CHART_PAGES 페이지까지 수집(50×4=200).
    # videos.list는 호출당 1유닛뿐이라 국가당 최대 4호출(총 ~60유닛)로 쿼터 부담 없음.
    max_pages = int(getattr(config, "YOUTUBE_CHART_PAGES", 4))
    for country in config.TARGET_COUNTRIES:
        items: list[dict] = []
        page_token = None
        err_status = None
        for _page in range(max_pages):
            params = {
                "part": "snippet,statistics",
                "chart": "mostPopular",
                "regionCode": country,
                "maxResults": 50,
            }
            if page_token:
                params["pageToken"] = page_token
            response = _yt_get(videos_endpoint, params)
            if response.status_code >= 400:
                err_status = response.status_code
                break
            payload = response.json()
            items.extend(payload.get("items", []))
            page_token = payload.get("nextPageToken")
            time.sleep(0.12)
            if not page_token:
                break
        if not items and err_status is not None:
            for genre in config.GENRE_WEIGHTS:
                rows.append(
                    {
                        "country": country,
                        "genre": genre,
                        "youtube_views": 0,
                        "youtube_matched_videos": 0,
                        "youtube_error": f"http_{err_status}",
                        "source": "youtube_api_error",
                    }
                )
            continue
        for genre in config.GENRE_WEIGHTS:
            total_views = 0
            matched = 0
            for item in items:
                snippet = item.get("snippet", {})
                title = snippet.get("title", "")
                channel = snippet.get("channelTitle", "")
                description = snippet.get("description", "")
                if _text_matches_genre(f"{title} {channel} {description}", genre):
                    total_views += int(item.get("statistics", {}).get("viewCount", 0))
                    matched += 1
            source = "youtube_api"
            error = ""
            if matched == 0 and config.YOUTUBE_SEARCH_FALLBACK:
                key = f"{country}|{genre}"
                hit = cache.get(key)
                fresh = hit and (
                    datetime.strptime(_yt_today, "%Y-%m-%d")
                    - datetime.strptime(hit.get("date", "1970-01-01"), "%Y-%m-%d")
                ).days < YT_CACHE_DAYS
                if fresh:
                    total_views = hit["views"]
                    matched = hit["matched"]
                    source = "youtube_search_cache"
                elif _yt_state["search_calls"] < YT_SEARCH_BUDGET:
                    _yt_state["search_calls"] += 1
                    fallback = _collect_youtube_search_metric(country, genre)
                    if fallback["youtube_matched_videos"] > 0:
                        total_views = fallback["youtube_views"]
                        matched = fallback["youtube_matched_videos"]
                        source = "youtube_search_fallback"
                        cache[key] = {"date": _yt_today, "views": total_views,
                                      "matched": matched}
                    elif fallback["youtube_error"]:
                        error = fallback["youtube_error"]
                        total_views = float("nan")
                else:
                    # 예산 소진. 0 으로 채우지 않고 결측으로 남긴다.
                    _yt_state["skipped"] += 1
                    error = "search_budget_exhausted"
                    total_views = float("nan")
                    source = "youtube_budget_skip"
            rows.append(
                {
                    "country": country,
                    "genre": genre,
                    "youtube_views": total_views,
                    "youtube_matched_videos": matched,
                    "youtube_error": error,
                    "source": source,
                }
            )
    _yt_cache_save(cache)
    print(f"[youtube] search 호출 {_yt_state['search_calls']}/{YT_SEARCH_BUDGET} "
          f"(≈{_yt_state['search_calls'] * 100} units), "
          f"예산 소진으로 건너뜀 {_yt_state['skipped']}건")
    return pd.DataFrame(rows)


def _collect_youtube_search_metric(country: str, genre: str) -> dict[str, int | str]:
    search_endpoint = "https://www.googleapis.com/youtube/v3/search"
    videos_endpoint = "https://www.googleapis.com/youtube/v3/videos"
    search_params = {
        "part": "snippet",
        "q": config.YOUTUBE_SEARCH_QUERIES[genre],
        "type": "video",
        "regionCode": country,
        "maxResults": config.YOUTUBE_SEARCH_MAX_RESULTS,
    }
    search_response = _yt_get(search_endpoint, search_params)
    if search_response.status_code >= 400:
        return {"youtube_views": 0, "youtube_matched_videos": 0, "youtube_error": f"search_http_{search_response.status_code}"}
    video_ids = [
        item.get("id", {}).get("videoId")
        for item in search_response.json().get("items", [])
        if item.get("id", {}).get("videoId")
    ]
    if not video_ids:
        return {"youtube_views": 0, "youtube_matched_videos": 0, "youtube_error": ""}

    stats_response = _yt_get(
        videos_endpoint,
        {"part": "statistics", "id": ",".join(video_ids)},
    )
    if stats_response.status_code >= 400:
        return {"youtube_views": 0, "youtube_matched_videos": 0, "youtube_error": f"stats_http_{stats_response.status_code}"}
    total_views = sum(int(item.get("statistics", {}).get("viewCount", 0)) for item in stats_response.json().get("items", []))
    return {"youtube_views": total_views, "youtube_matched_videos": len(video_ids), "youtube_error": ""}


def _sample_youtube_metrics() -> pd.DataFrame:
    rows = []
    multipliers = {
        "US": 1.00, "CN": 0.52, "JP": 0.64, "VN": 0.78, "TH": 0.58,
        "ID": 0.82, "IN": 0.70, "MY": 0.54, "FR": 0.48, "GB": 0.50,
        "BR": 0.62, "AR": 0.44, "AE": 0.40, "TR": 0.43, "ZA": 0.32,
    }
    genre_base = {
        "kpop": 3200000, "kvideo": 1900000, "kgame": 1500000, "kwebtoon": 700000,
        "kfood": 620000, "kbeauty": 880000, "kfashion": 540000, "ktourism": 410000,
    }
    for country, country_mul in multipliers.items():
        for genre, base in genre_base.items():
            rows.append(
                {
                    "country": country,
                    "genre": genre,
                    "youtube_views": int(base * country_mul * (0.88 + len(country + genre) % 7 / 25)),
                    "youtube_matched_videos": 3 + (len(country + genre) % 8),
                    "youtube_error": "",
                    "source": "sample_youtube",
                }
            )
    return pd.DataFrame(rows)


def _reddit_token() -> str:
    auth = f"{config.REDDIT_CLIENT_ID}:{config.REDDIT_CLIENT_SECRET}".encode("utf-8")
    headers = {
        "Authorization": "Basic " + base64.b64encode(auth).decode("ascii"),
        "User-Agent": "kwci_pipeline/1.0 by local-research",
    }
    response = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        data={"grant_type": "client_credentials"},
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def collect_reddit_metrics(sample: bool = False, days_back: int = 7) -> pd.DataFrame:
    if sample or not (config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET):
        return _sample_reddit_metrics()

    token = _reddit_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "kwci_pipeline/1.0 by local-research",
    }
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    rows = []
    for country, meta in config.TARGET_COUNTRIES.items():
        terms = [term.lower() for term in meta["reddit_terms"]]
        for genre, subreddits in config.GENRE_SUBREDDITS.items():
            if country in config.REDDIT_RESTRICTED:
                rows.append(
                    {
                        "country": country,
                        "genre": genre,
                        "reddit_mentions": 0,
                        "reddit_restricted": True,
                        "source": "reddit_restricted",
                    }
                )
                continue
            mentions = 0
            for subreddit in subreddits:
                url = f"https://oauth.reddit.com/r/{subreddit}/new"
                response = requests.get(url, params={"limit": 100}, headers=headers, timeout=30)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                for child in response.json().get("data", {}).get("children", []):
                    post = child.get("data", {})
                    created = datetime.fromtimestamp(post.get("created_utc", 0), tz=timezone.utc)
                    if created < cutoff:
                        continue
                    text = f"{post.get('title', '')} {post.get('selftext', '')}".lower()
                    if any(term in text for term in terms):
                        mentions += 1 + int(post.get("num_comments", 0))
                time.sleep(0.18)
            rows.append(
                {
                    "country": country,
                    "genre": genre,
                    "reddit_mentions": mentions,
                    "reddit_restricted": False,
                    "source": "reddit_api",
                }
            )
    return pd.DataFrame(rows)


def _sample_reddit_metrics() -> pd.DataFrame:
    rows = []
    base = {
        "US": 18500, "CN": 0, "JP": 2700, "VN": 900, "TH": 1200,
        "ID": 2100, "IN": 3400, "MY": 800, "FR": 1900, "GB": 3100,
        "BR": 1600, "AR": 700, "AE": 420, "TR": 620, "ZA": 350,
    }
    genre_mul = {
        "kpop": 1.0, "kvideo": 0.60, "kgame": 0.45, "kwebtoon": 0.30,
        "kfood": 0.15, "kbeauty": 0.28, "kfashion": 0.14, "ktourism": 0.11,
    }
    for country, country_base in base.items():
        for genre, mul in genre_mul.items():
            rows.append(
                {
                    "country": country,
                    "genre": genre,
                    "reddit_mentions": int(country_base * mul),
                    "reddit_restricted": country in config.REDDIT_RESTRICTED,
                    "source": "sample_reddit",
                }
            )
    return pd.DataFrame(rows)


def collect_kf_metrics(sample: bool = False) -> pd.DataFrame:
    if sample or not (config.KF_API_KEY and config.KF_API_URL):
        return _sample_kf_metrics()

    rows: list[dict[str, Any]] = []
    for country, meta in config.TARGET_COUNTRIES.items():
        response = requests.get(
            config.KF_API_URL,
            params={
                "ServiceKey": config.KF_API_KEY,
                "pageNo": "1",
                "numOfRows": "100",
                "returnType": "JSON",
                "cond[country_iso_alp2::EQ]": country,
            },
            timeout=30,
        )
        if response.status_code in {401, 403}:
            print(f"[kwci] KF API unauthorized for {country}: using sample KF metrics.")
            return _sample_kf_metrics()
        if response.status_code >= 400:
            rows.append({"country": country, "kf_count": 0, "source": f"kf_api_error_{response.status_code}"})
            continue
        payload = response.json()
        count = _extract_count(payload)
        rows.append({"country": country, "kf_count": count, "source": "kf_api"})
        time.sleep(0.1)
    return pd.DataFrame(rows)


def _extract_count(payload: Any) -> int:
    if isinstance(payload, dict):
        body = payload.get("response", {}).get("body", {})
        total = body.get("totalCount") or body.get("total_count")
        if total is not None:
            try:
                return int(total)
            except (TypeError, ValueError):
                pass
        items = body.get("items", {})
        if isinstance(items, dict) and isinstance(items.get("item"), list):
            return len(items["item"])
        if isinstance(items, dict) and isinstance(items.get("item"), dict):
            return 1
        counts = [_extract_count(value) for value in payload.values()]
        return max(counts, default=0)
    if isinstance(payload, list):
        return len(payload)
    return 0


def _sample_kf_metrics() -> pd.DataFrame:
    rows = []
    for country, values in config.SAMPLE_SURVEY.items():
        rows.append({"country": country, "kf_count": values["kf"], "source": "sample_kf"})
    return pd.DataFrame(rows)


def collect_trends(sample: bool = False) -> pd.DataFrame:
    """Google Trends(pytrends) → 분야×국가 검색 관심도(0~100). L3 수용자 신호.

    분야당 1회 interest_by_region(COUNTRY) 호출로 전 국가 관심도 수신 → 15개국 추출.
    pytrends 미설치/오류 시 샘플(설문 기반) 대체. (pip install pytrends)
    """
    if sample:
        return _sample_trends()
    try:
        from pytrends.request import TrendReq
    except Exception:  # noqa: BLE001
        return _sample_trends()
    try:
        py = TrendReq(hl="en-US", tz=0)
    except Exception:  # noqa: BLE001
        return _sample_trends()
    rows, any_real = [], False
    for genre, query in config.TRENDS_QUERIES.items():
        reg = None
        try:
            py.build_payload([query], timeframe="today 3-m")
            reg = py.interest_by_region(resolution="COUNTRY", inc_low_vol=True)
            time.sleep(1.0)
        except Exception:  # noqa: BLE001
            reg = None
        if reg is None or reg.empty:
            for c in config.TARGET_COUNTRIES:
                rows.append({"country": c, "genre": genre,
                             "trends_interest": _sample_trend_value(c, genre), "source": "sample_trends"})
            continue
        any_real = True
        col = reg.columns[0]
        for code, name in config.TRENDS_GEO_NAME.items():
            val = 0.0
            if name in reg.index:
                val = float(reg.loc[name, col])
            elif name == "Turkey" and "Türkiye" in reg.index:
                val = float(reg.loc["Türkiye", col])
            rows.append({"country": code, "genre": genre, "trends_interest": val, "source": "trends_api"})
    return pd.DataFrame(rows) if (rows and any_real) else _sample_trends()


def _sample_trend_value(country: str, genre: str) -> float:
    return float(config.SAMPLE_SURVEY.get(country, {}).get(genre, 40))


def _sample_trends() -> pd.DataFrame:
    rows = []
    for c in config.TARGET_COUNTRIES:
        for g in config.GENRE_WEIGHTS:
            rows.append({"country": c, "genre": g,
                         "trends_interest": _sample_trend_value(c, g), "source": "sample_trends"})
    return pd.DataFrame(rows)


def _parse_customs_expdlr(xml_text: str):
    """관세청 응답(XML)에서 expDlr(수출금액 USD) 합산.

    반환: (합계, 항목수, 결과코드)

    **왜 항목수를 같이 돌려주는가.** 이전 구현은 <expDlr> 태그가 하나도 없으면
    조용히 0.0 을 돌려주었다. 그래서 "아직 공표되지 않은 달"과 "수출이 실제로 0"
    이 구분되지 않았고, 2026-07-02 수집분 45행이 전부 0 이 되었는데도 오류
    플래그가 붙지 않았다. 그 0 이 정규화를 거쳐 전 국가 50.0 이 되었다.
    """
    items = re.findall(r"<expDlr>([0-9.]+)</expDlr>", xml_text)
    code = re.search(r"<resultCode>([^<]*)</resultCode>", xml_text)
    msg = re.search(r"<(?:returnAuthMsg|errMsg|resultMsg)>([^<]*)<", xml_text)
    rc = (code.group(1).strip() if code else "") or (msg.group(1).strip() if msg else "")
    return sum(float(x) for x in items if x), len(items), rc


def collect_customs_export(sample: bool = False) -> pd.DataFrame:
    """관세청 품목별 국가별 수출입실적 → 분야별 L1 경제(수출액 USD).

    kfood/kfashion/kbeauty만 HS 코드 기반으로 국가별 수출액을 모은다(직전 완전월 1개월).
    키 없거나 오류면 샘플로 대체.
    """
    if sample or not config.DATA_GO_KR_API_KEY:
        return _sample_customs_export()

    now = datetime.now(timezone(timedelta(hours=9)))
    # **기준월을 2개월 전으로 잡는다.** 관세청 무역통계는 익월 중순에 확정
    # 공표되므로, 직전월을 요청하면 월초 실행에서 빈 응답이 온다. 실제로
    # 06-28 실행(→2026-05)은 19억 달러가 들어왔고 07-02 실행(→2026-06)은 0 이었다.
    first = now.replace(day=1)
    ref = (first - timedelta(days=1)).replace(day=1) - timedelta(days=1)
    yymm = ref.strftime("%Y%m")
    rows = []
    for genre, hs_list in config.CUSTOMS_HS_CODES.items():
        for country in config.TARGET_COUNTRIES:
            total, err, nitem = 0.0, "", 0
            for hs in hs_list:
                try:
                    r = requests.get(config.CUSTOMS_EXPORT_URL, params={
                        "serviceKey": config.DATA_GO_KR_API_KEY,
                        "strtYymm": yymm, "endYymm": yymm,
                        "hsSgn": hs, "cntyCd": country,
                    }, timeout=30)
                except Exception:  # noqa: BLE001
                    err = "network"
                    continue
                if r.status_code >= 400:
                    err = f"http_{r.status_code}"
                    continue
                sub, cnt, rc = _parse_customs_expdlr(r.text)
                total += sub
                nitem += cnt
                if cnt == 0 and rc and rc not in {"00", "0", "NORMAL SERVICE."}:
                    err = f"api:{rc[:40]}"
                time.sleep(0.08)
            # 항목이 하나도 없으면 "수출 0" 이 아니라 **결측**이다.
            # 0 을 내보내면 정규화가 그것을 실측으로 취급한다.
            if nitem == 0:
                rows.append({"country": country, "genre": genre,
                             "export_usd": float("nan"),
                             "customs_error": err or f"no_items({yymm})",
                             "source": "customs_api_empty"})
            else:
                rows.append({"country": country, "genre": genre, "export_usd": total,
                             "customs_error": err,
                             "source": "customs_api" if not err else "customs_api_partial"})
    if not rows:
        return _sample_customs_export()
    df = pd.DataFrame(rows)
    # 전량 결측이면 조용히 넘어가지 않는다. 이 상태로 지수를 만들면 안 된다.
    if df["export_usd"].notna().sum() == 0:
        raise RuntimeError(
            f"관세청 수집 전량 실패 (기준월 {yymm}). 지수 산출을 중단한다. "
            f"사유 예: {df['customs_error'].iloc[0]}")
    return df


def _comtrade_fetch(period: str, cmd: str):
    """Comtrade 한 달치. 실패는 None, 빈 응답은 []."""
    params = {"reporterCode": "410", "period": period, "flowCode": "X",
              "cmdCode": cmd, "partner2Code": "0", "customsCode": "C00", "motCode": "0"}
    for attempt in range(5):
        try:
            r = requests.get(config.COMTRADE_URL, params=params,
                             headers={"User-Agent": _COMTRADE_UA}, timeout=60)
        except Exception:  # noqa: BLE001
            time.sleep(4 * (attempt + 1)); continue
        if r.status_code == 200:
            return r.json().get("data") or []
        if r.status_code in (429, 500, 502, 503):
            time.sleep(6 * (attempt + 1)); continue
        return None
    return None


def collect_comtrade_export(sample: bool = False) -> pd.DataFrame:
    """UN Comtrade → 도메인별·국가별 L1 수출액 (키 불필요).

    관세청 API 를 대체하는 1차 경로다. 공표 시차가 2~9개월이라 고정 오프셋을
    쓰면 빈 응답이 온다 — 2026-07-02 수집 사고가 정확히 그것이었다.
    여기서는 최신 월부터 뒤로 훑어 자료가 있는 달을 찾고, 그 달을 기록한다.
    """
    if sample:
        return _sample_customs_export()
    now = datetime.now(timezone(timedelta(hours=9)))
    rows, used = [], {}

    # 기준월은 한 번만 찾는다. 도메인마다 훑으면 요청이 4배가 되고,
    # 공표 시차는 품목과 무관하게 같다. 레코드가 많은 화장품으로 탐색한다.
    probe_cmd = config.COMTRADE_HS["kbeauty"]
    ref_period, cursor = None, now.replace(day=1)
    for _ in range(config.COMTRADE_LOOKBACK):
        cursor = (cursor - timedelta(days=1)).replace(day=1)
        period = cursor.strftime("%Y%m")
        if _comtrade_fetch(period, probe_cmd):
            ref_period = period
            break
        time.sleep(1.3)
    if ref_period is None:
        raise RuntimeError(
            f"Comtrade 최신 공표월을 {config.COMTRADE_LOOKBACK}개월 안에서 찾지 못했다.")
    print(f"[comtrade] 최신 공표월 = {ref_period}")

    for genre, cmd in config.COMTRADE_HS.items():
        data = _comtrade_fetch(ref_period, cmd)
        time.sleep(1.3)
        if not data:
            for cc in config.TARGET_COUNTRIES:
                rows.append({"country": cc, "genre": genre, "export_usd": float("nan"),
                             "customs_error": f"comtrade_no_data({ref_period})",
                             "source": "comtrade_empty"})
            continue
        used[genre] = ref_period
        agg: dict[str, float] = {}
        for item in data:
            cc = config.COMTRADE_M49.get(item.get("partnerCode"))
            val = item.get("primaryValue")
            if cc and cc in config.TARGET_COUNTRIES and val:
                agg[cc] = agg.get(cc, 0.0) + float(val)
        for cc in config.TARGET_COUNTRIES:
            rows.append({"country": cc, "genre": genre,
                         "export_usd": agg.get(cc, float("nan")),
                         "customs_error": "" if cc in agg else f"not_in_comtrade({ref_period})",
                         "source": "comtrade"})
    df = pd.DataFrame(rows)
    if df["export_usd"].notna().sum() == 0:
        raise RuntimeError("Comtrade 수집 전량 실패. 지수 산출을 중단한다.")
    print(f"[comtrade] 기준월 {used} · 유효 관측 "
          f"{int(df['export_usd'].notna().sum())}/{len(df)}")
    return df


def _sample_customs_export() -> pd.DataFrame:
    base = {"kfood": 5_000_000, "kfashion": 4_000_000, "kbeauty": 8_000_000}
    cmul = {"US": 1.0, "CN": 1.4, "JP": 1.2, "VN": 0.5, "TH": 0.45, "ID": 0.4,
            "IN": 0.3, "MY": 0.35, "FR": 0.5, "GB": 0.5, "BR": 0.4, "AR": 0.2,
            "AE": 0.3, "TR": 0.25, "ZA": 0.15}
    rows = []
    for genre, b in base.items():
        for c, m in cmul.items():
            rows.append({"country": c, "genre": genre,
                         "export_usd": int(b * m * (0.85 + len(c + genre) % 6 / 20)),
                         "customs_error": "", "source": "sample_customs"})
    return pd.DataFrame(rows)


def collect_tourism_supply(sample: bool = False) -> pd.DataFrame:
    """TourAPI KorService2 → 전국 관광 공급 인프라 스냅샷(콘텐츠 타입별 총개수).

    국가별이 아니라 전국 단위 공급 지표. DATA_GO_KR_API_KEY 사용. 키 없거나 오류면 샘플.
    """
    if sample or not config.DATA_GO_KR_API_KEY:
        return _sample_tourism_supply()
    rows = []
    for ctid, name in config.TOURAPI_CONTENT_TYPES.items():
        total, src = 0, "tourapi"
        try:
            r = requests.get(config.TOURAPI_URL, params={
                "serviceKey": config.DATA_GO_KR_API_KEY, "MobileOS": "ETC", "MobileApp": "kwci",
                "_type": "json", "numOfRows": 1, "pageNo": 1, "contentTypeId": ctid}, timeout=30)
            total = int(r.json()["response"]["body"]["totalCount"])
        except Exception:  # noqa: BLE001
            src = "tourapi_error"
        rows.append({"content_type_id": ctid, "content_type_name": name,
                     "total_count": total, "source": src})
        time.sleep(0.1)
    if all(x["total_count"] == 0 for x in rows):
        return _sample_tourism_supply()
    return pd.DataFrame(rows)


def _sample_tourism_supply() -> pd.DataFrame:
    base = {12: 17000, 14: 3000, 15: 5000, 25: 1000, 28: 3000, 32: 30000, 38: 12000, 39: 40000}
    return pd.DataFrame([{"content_type_id": k, "content_type_name": config.TOURAPI_CONTENT_TYPES[k],
                          "total_count": v, "source": "sample_tourapi"} for k, v in base.items()])


def collect_kosis_industry(sample: bool = False) -> pd.DataFrame:
    """KOSIS 콘텐츠산업조사 수출액(orgId=113) → 분야별 전국 연간 수출(천달러).

    음악→kpop, 방송+영화→kvideo, 게임→kgame, 만화→kwebtoon. 전국·연 단위(도메인 L1 컨텍스트).
    KOSIS_API_KEY 없거나 오류면 샘플(2023 실측 기반).
    """
    if sample or not config.KOSIS_API_KEY:
        return _sample_kosis_industry()
    try:
        r = requests.get(config.KOSIS_URL, params={
            "method": "getList", "apiKey": config.KOSIS_API_KEY, "orgId": config.KOSIS_ORG_ID,
            "tblId": config.KOSIS_EXPORT_TBL, "itmId": config.KOSIS_EXPORT_ITM, "objL1": "ALL",
            "prdSe": "Y", "newEstPrdCnt": "1", "format": "json", "jsonVD": "Y"}, timeout=30)
        data = r.json()
    except Exception:  # noqa: BLE001
        return _sample_kosis_industry()
    if not isinstance(data, list) or not data:
        return _sample_kosis_industry()
    agg, year = {}, None
    for rec in data:
        genre = config.KOSIS_INDUSTRY_MAP.get((rec.get("C1_NM") or "").strip())
        if not genre:
            continue
        try:
            val = float(rec.get("DT") or 0)
        except (TypeError, ValueError):
            val = 0.0
        agg[genre] = agg.get(genre, 0.0) + val
        year = rec.get("PRD_DE", year)
    if not agg:
        return _sample_kosis_industry()
    return pd.DataFrame([{"genre": g, "export_kusd": v, "year": year, "source": "kosis_api"}
                         for g, v in agg.items()])


def _sample_kosis_industry() -> pd.DataFrame:
    s = {"kpop": 1_220_000, "kvideo": 1_106_000, "kgame": 8_390_000, "kwebtoon": 178_000}
    return pd.DataFrame([{"genre": g, "export_kusd": v, "year": "2023", "source": "sample_kosis"}
                         for g, v in s.items()])


def collect_kto_visitors(sample: bool = False, ym: str | None = None) -> pd.DataFrame:
    """KTO 출입국관광통계(15000297) → 국가별 방한 외래관광객 수(월). ktourism의 per-country L1.

    NAT_CD 코드 대신 응답의 국가명(natKorNm)을 우리 국가코드에 매칭(공백 제거). XML 응답.
    export_usd 컬럼으로 내보내 관세청 수출과 동일 L1 경로로 흐른다(분야별 정규화이므로 단위 무관).
    """
    if sample or not config.DATA_GO_KR_API_KEY:
        return _sample_kto_visitors()
    if ym is None:
        # 관세청과 같은 이유로 2개월 전을 쓴다. 출입국통계도 익월 공표다.
        _now = datetime.now(timezone(timedelta(hours=9)))
        _first = _now.replace(day=1)
        ym = ((_first - timedelta(days=1)).replace(day=1) - timedelta(days=1)).strftime("%Y%m")
    try:
        r = requests.get(config.KTO_VISITORS_URL, params={
            "ServiceKey": config.DATA_GO_KR_API_KEY, "YM": ym, "ED_CD": "E",
            "numOfRows": 400, "pageNo": 1}, timeout=30)
        text = r.text
    except Exception:  # noqa: BLE001
        return _sample_kto_visitors()
    rows, unmatched, n_items = [], [], 0
    for m in re.finditer(r"<item>(.*?)</item>", text, re.S):
        blob = m.group(1)
        nm = re.search(r"<natKorNm>(.*?)</natKorNm>", blob)
        num = re.search(r"<num>(\d+)</num>", blob)
        if not nm or not num:
            continue
        n_items += 1
        raw = nm.group(1).replace(" ", "").strip()
        code = config.KTO_NAME_MAP.get(raw)
        if not code:
            unmatched.append(raw)
            continue
        rows.append({"country": code, "genre": "ktourism", "export_usd": float(num.group(1)),
                     "customs_error": "", "source": "kto_api"})
    # **합성 샘플로 조용히 대체하지 않는다.** 이전 구현은 한 건도 매칭되지 않으면
    # sample_kto 를 돌려주었고, 그 합성값이 K관광 L1(가중 0.5)에 그대로 들어갔다.
    if not rows:
        raise RuntimeError(
            f"KTO 수집 실패 (기준월 {ym}, 응답 항목 {n_items}건). "
            f"매칭 안 된 국가명 예: {unmatched[:8]}. "
            "KTO_NAME_MAP 을 응답의 natKorNm 표기에 맞춰야 한다.")
    missing = sorted(set(config.TARGET_COUNTRIES) - {r["country"] for r in rows})
    for c in missing:
        rows.append({"country": c, "genre": "ktourism", "export_usd": float("nan"),
                     "customs_error": f"not_in_kto_response({ym})",
                     "source": "kto_api_missing"})
    if unmatched:
        print(f"[kto] 매칭 안 된 국가명 {len(set(unmatched))}종: {sorted(set(unmatched))[:12]}")
    return pd.DataFrame(rows)


def _sample_kto_visitors() -> pd.DataFrame:
    base = {"US": 120000, "CN": 300000, "JP": 250000, "VN": 50000, "TH": 40000, "ID": 35000,
            "IN": 30000, "MY": 30000, "FR": 25000, "GB": 25000, "BR": 15000, "AR": 8000,
            "AE": 6000, "TR": 7000, "ZA": 5000}
    return pd.DataFrame([{"country": c, "genre": "ktourism", "export_usd": float(v),
                          "customs_error": "", "source": "sample_kto"} for c, v in base.items()])


def save_raw(df: pd.DataFrame, name: str) -> Path:
    ensure_dirs()
    path = config.RAW_DIR / f"{now_kst_label()}_{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def append_timeseries(df: pd.DataFrame, name: str) -> Path:
    """소스별 시계열 누적 저장: data/timeseries/{name}.csv 에 collected_at 붙여 append."""
    ts_dir = config.DATA_DIR / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)
    path = ts_dir / f"{name}.csv"
    out = df.copy()
    out.insert(0, "collected_at", now_kst_label())
    out.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")
    return path
