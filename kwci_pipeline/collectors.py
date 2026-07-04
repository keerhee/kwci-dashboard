from __future__ import annotations

import base64
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from . import config


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


def collect_youtube_metrics(sample: bool = False) -> pd.DataFrame:
    if sample or not config.YOUTUBE_API_KEY:
        return _sample_youtube_metrics()

    rows = []
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
                fallback = _collect_youtube_search_metric(country, genre)
                if fallback["youtube_matched_videos"] > 0:
                    total_views = fallback["youtube_views"]
                    matched = fallback["youtube_matched_videos"]
                    source = "youtube_search_fallback"
                elif fallback["youtube_error"]:
                    error = fallback["youtube_error"]
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


def _parse_customs_expdlr(xml_text: str) -> float:
    """관세청 응답(XML)에서 expDlr(수출금액 USD) 합산."""
    return sum(float(x) for x in re.findall(r"<expDlr>([0-9.]+)</expDlr>", xml_text) if x)


def collect_customs_export(sample: bool = False) -> pd.DataFrame:
    """관세청 품목별 국가별 수출입실적 → 분야별 L1 경제(수출액 USD).

    kfood/kfashion/kbeauty만 HS 코드 기반으로 국가별 수출액을 모은다(직전 완전월 1개월).
    키 없거나 오류면 샘플로 대체.
    """
    if sample or not config.DATA_GO_KR_API_KEY:
        return _sample_customs_export()

    now = datetime.now(timezone(timedelta(hours=9)))
    last_month = now.replace(day=1) - timedelta(days=1)
    yymm = last_month.strftime("%Y%m")
    rows = []
    for genre, hs_list in config.CUSTOMS_HS_CODES.items():
        for country in config.TARGET_COUNTRIES:
            total, err = 0.0, ""
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
                total += _parse_customs_expdlr(r.text)
                time.sleep(0.08)
            rows.append({"country": country, "genre": genre, "export_usd": total,
                         "customs_error": err,
                         "source": "customs_api" if not err else "customs_api_partial"})
    if not rows:
        return _sample_customs_export()
    return pd.DataFrame(rows)


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
        lm = datetime.now(timezone(timedelta(hours=9))).replace(day=1) - timedelta(days=1)
        ym = lm.strftime("%Y%m")
    try:
        r = requests.get(config.KTO_VISITORS_URL, params={
            "ServiceKey": config.DATA_GO_KR_API_KEY, "YM": ym, "ED_CD": "E",
            "numOfRows": 400, "pageNo": 1}, timeout=30)
        text = r.text
    except Exception:  # noqa: BLE001
        return _sample_kto_visitors()
    rows = []
    for m in re.finditer(r"<item>(.*?)</item>", text, re.S):
        blob = m.group(1)
        nm = re.search(r"<natKorNm>(.*?)</natKorNm>", blob)
        num = re.search(r"<num>(\d+)</num>", blob)
        if not nm or not num:
            continue
        code = config.KTO_NAME_MAP.get(nm.group(1).replace(" ", "").strip())
        if not code:
            continue
        rows.append({"country": code, "genre": "ktourism", "export_usd": float(num.group(1)),
                     "customs_error": "", "source": "kto_api"})
    return pd.DataFrame(rows) if rows else _sample_kto_visitors()


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
