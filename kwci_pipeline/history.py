"""분기 시계열 엔진 — 2018Q1~현재, 기준연도 2018=100. (실데이터 역사 백필)

경제(L1) 백본 지수의 분기별 추이. 백필 가능한 소스는 실데이터로, 나머지는 추세 샘플로 채운다.
도메인별 데이터 출처를 sources에 표기(real vs sample)해 정직하게 구분한다.

실 백필:
  - 콘텐츠 4분야(음악·방송/영화·게임·만화)는 확정된 연간 공식통계 기반 검증 고정값(2018=100)을
    결정론적으로 사용한다. 연간 수출은 확정되면 바뀌지 않으므로 실행마다 재호출하지 않는다.
    (KOSIS 콘텐츠산업조사 '수출' 라이브 합산은 방송·만화 항목이 알려진 실적과 크게 어긋나 표시
     시계열에는 채택하지 않는다. 진단용 _kosis_annual()은 유지하되 표시 계열은 고정값 사용.)
  - KOSIS 국적·지역별 외국인 입국자(orgId=111, DT_091_111_2009_S005A): 연간 2018~2024 →
    ktourism 볼륨(2018=100) + source-market 다각화. (KOSIS_API_KEY 필요. 2025는 공표연간.)
  - 관세청(kfood·kfashion·kbeauty): 무역통계 연간 수출(가공식품·의류·화장품 HS) 라이브.
"""
from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from kwci_pipeline import config, processor, collectors
else:
    from . import config, processor, collectors

GROWTH = {"kpop": 0.08, "kvideo": 0.12, "kgame": 0.06, "kwebtoon": 0.20,
          "kfood": 0.09, "kbeauty": 0.11, "kfashion": 0.04, "ktourism": 0.10}
TOURISM_COVID = {
    (2020, 1): 0.85, (2020, 2): 0.05, (2020, 3): 0.04, (2020, 4): 0.04,
    (2021, 1): 0.05, (2021, 2): 0.06, (2021, 3): 0.08, (2021, 4): 0.12,
    (2022, 1): 0.18, (2022, 2): 0.30, (2022, 3): 0.45, (2022, 4): 0.60,
    (2023, 1): 0.72, (2023, 2): 0.82, (2023, 3): 0.90, (2023, 4): 0.93,
    (2024, 1): 0.95, (2024, 2): 0.98, (2024, 3): 1.00, (2024, 4): 1.00,
}

# ── 실측 연간 지수(2018=100) ───────────────────────────────────
# 각 도메인 L1 경제지표(수출/매출)의 검증 통계 기반 연간 지수. 엔드포인트(2018·2024)는
# 공식 통계로 검증, 중간연도는 실측/근사. 출처: 콘텐츠산업조사(음악·방송·게임), KOCCA
# 웹툰 실태조사(매출), 농식품부(농식품 수출), 식약처(화장품 수출), KOSIS 국적별 외래객.
#   음악 5.64→18.0억$ / 방송 4.78→12.52억$ / 게임 64.1→85.0억$ / 웹툰매출 4663→22856억원
#   농식품 64.8→99.8억$ / 화장품 62.8→102억$ / 의류 보합
#   외래객(총계, KOSIS orgId=111): 2018 1,563만→2024 1,697만명(2018=100→108.6), 2025 공표 약 1,870만(122)
REAL_ANNUAL_INDEX = {
    "kpop":     {2018: 100, 2019: 135, 2020: 121, 2021: 154, 2022: 160, 2023: 230, 2024: 319},
    "kvideo":   {2018: 100, 2019: 99,  2020: 145, 2021: 147, 2022: 167, 2023: 188, 2024: 262},
    "kgame":    {2018: 100, 2019: 104, 2020: 128, 2021: 135, 2022: 139, 2023: 117, 2024: 133},
    "kwebtoon": {2018: 100, 2019: 137, 2020: 226, 2021: 336, 2022: 392, 2023: 469, 2024: 490},
    "kfood":    {2018: 100, 2019: 108, 2020: 117, 2021: 132, 2022: 136, 2023: 139, 2024: 154},
    "kbeauty":  {2018: 100, 2019: 104, 2020: 121, 2021: 146, 2022: 127, 2023: 135, 2024: 162},
    "kfashion": {2018: 100, 2019: 98,  2020: 87,  2021: 100, 2022: 108, 2023: 104, 2024: 105},
    # 관광: KOSIS 국적별 외래객 총계(orgId=111) 실측 2018=100. 2025는 공표 연간(대시보드 기존값) 유지.
    "ktourism": {2018: 100, 2019: 114, 2020: 17,  2021: 7,   2022: 22,  2023: 74,  2024: 109, 2025: 122},
}
REAL_LAST_YEAR = 2025  # 분야별로 자체 최신연도를 쓰며, 시계열 표시는 2025까지 캡

# ── 관광 source-market 국적별 비중(%) — 다각화 지수용 ──────────────────────
# 출처: KOSIS 국적·지역별 외국인 입국자(orgId=111, DT_091_111_2009_S005A) 2018·2020–2024 실측.
# 각 연도 총계 대비 상위 명명시장(중·일·대만·미·홍콩) 비중, 잔여는 'others'.
# 2019는 국적별 분해가 표에 비어 공표 연간통계로 보정, 2025는 공표 연간(대시보드 기존값).
# 사드(2017)·코로나(2020–22)를 거치며 중국 의존이 일시 완화되었다가 2024년 재집중(27%).
TOURISM_SHARES = {
    2018: {"CN": 25.5, "JP": 19.0, "TW": 7.3, "US": 6.8, "HK": 4.4, "others": 37.0},
    2019: {"CN": 34.4, "JP": 18.7, "TW": 7.2, "US": 6.2, "HK": 4.0, "others": 29.5},
    2020: {"CN": 22.9, "JP": 16.5, "TW": 6.5, "US": 10.0, "HK": 3.3, "others": 40.8},
    2021: {"CN": 11.7, "JP": 1.7,  "TW": 0.6, "US": 24.0, "HK": 0.1, "others": 61.9},
    2022: {"CN": 6.1,  "JP": 9.1,  "TW": 2.3, "US": 18.0, "HK": 1.8, "others": 62.7},
    2023: {"CN": 17.0, "JP": 20.4, "TW": 8.5, "US": 10.2, "HK": 3.5, "others": 40.4},
    2024: {"CN": 27.1, "JP": 19.2, "TW": 8.8, "US": 8.3, "HK": 3.3, "others": 33.3},
    2025: {"CN": 25.0, "JP": 19.0, "TW": 9.5, "US": 8.5, "HK": 3.5, "others": 34.5},
}


def _real_idx_q(genre: str, year: int, q: int) -> float:
    """실측 연간지수 → 분기지수. 연간값을 분기 중앙에 배치해 선형보간. 분야별 자체 최신연도까지 사용."""
    a = REAL_ANNUAL_INDEX[genre]
    last, first = max(a), min(a)

    def av(yr):
        if yr in a:
            return a[yr]
        return a[last] if yr > last else a[first]

    lo = av(year)
    hi = av(year + 1)
    return round(lo + (hi - lo) * (q - 0.5) / 4.0, 1)


def _idx_q_from_annual(aidx: dict, year: int, q: int) -> float:
    """이미 2018=100으로 지수화된 연간 dict(aidx)를 분기로 보간. 최신연도 이후는 유지."""
    last, first = max(aidx), min(aidx)

    def av(yr):
        if yr in aidx:
            return aidx[yr]
        if yr >= last:
            return aidx[last]
        if yr <= first:
            return aidx[first]
        lo_y = max(y for y in aidx if y < yr)
        hi_y = min(y for y in aidx if y > yr)
        return aidx[lo_y] + (aidx[hi_y] - aidx[lo_y]) * (yr - lo_y) / (hi_y - lo_y)

    lo = av(year)
    hi = av(year + 1)
    return round(lo + (hi - lo) * (q - 0.5) / 4.0, 1)


def _current_quarter():
    now = datetime.now(timezone(timedelta(hours=9)))
    return now.year, (now.month - 1) // 3 + 1


def quarters(base_year: int):
    ey, eq = _current_quarter()
    out, y, q = [], base_year, 1
    while (y, q) <= (ey, eq):
        out.append((y, q))
        q += 1
        if q > 4:
            y, q = y + 1, 1
    return out


# ── 샘플 추세 (폴백) ──────────────────────────────────────────
def _sample_domain_raw(genre: str, year: int, q: int) -> float:
    base = (1 + GROWTH[genre]) ** ((year - 2018) + (q - 1) / 4.0)
    if genre == "ktourism":
        base *= TOURISM_COVID.get((year, q), 1.0)
    seasonal = 1 + 0.03 * math.sin((q / 4.0) * 2 * math.pi)
    noise = 1 + ((hash((genre, year, q)) % 7) - 3) / 120.0
    return max(base * seasonal * noise, 0.001)


# ── 진단용: KOSIS 연간 (한 번 호출로 다년치) — 표시 계열엔 미사용 ──
def _kosis_annual():
    if not config.KOSIS_API_KEY:
        return None
    try:
        r = requests.get(config.KOSIS_URL, params={
            "method": "getList", "apiKey": config.KOSIS_API_KEY, "orgId": config.KOSIS_ORG_ID,
            "tblId": config.KOSIS_EXPORT_TBL, "itmId": config.KOSIS_EXPORT_ITM, "objL1": "ALL",
            "prdSe": "Y", "startPrdDe": str(config.HISTORY_BASE_YEAR),
            "endPrdDe": str(_current_quarter()[0]), "format": "json", "jsonVD": "Y"}, timeout=40)
        data = r.json()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, list) or not data:
        return None
    out = {}
    for rec in data:
        g = config.KOSIS_INDUSTRY_MAP.get((rec.get("C1_NM") or "").strip())
        if not g:
            continue
        try:
            y = int(rec.get("PRD_DE")); v = float(rec.get("DT") or 0)
        except (TypeError, ValueError):
            continue
        out.setdefault(g, {})
        out[g][y] = out[g].get(y, 0.0) + v
    return out or None


# ── 실 백필: KOSIS 국적별 외래객 (orgId=111) → 관광 볼륨 + 다각화 ──
KOSIS_INBOUND_TBL = "DT_091_111_2009_S005A"
KOSIS_INBOUND_ITM = "13103870964T1"
KOSIS_INBOUND_SEX_TOTAL = "13102870964B.0"  # 성별=계
KOSIS_INBOUND_MONTH_TOTAL = "C01"           # 월별=계(연간)
# 명명 상위시장 국적명(KOSIS C1_NM) → 코드
KOSIS_INBOUND_NAMED = {"중국": "CN", "일본": "JP", "타이완(대만)": "TW", "미국": "US", "홍콩": "HK"}
# 집계행(개별 국가 아님) — 다각화 ENM 계산에서 제외
KOSIS_INBOUND_AGG = ("총계",)


def _kosis_inbound_annual():
    """KOSIS 국적·지역별 외국인 입국자(orgId=111) 연간(성별·월 합계).

    반환: {"total": {year: 총계}, "by": {C1_NM: {year: 입국자}}}. 키 없거나 실패 시 None.
    """
    if not config.KOSIS_API_KEY:
        return None
    try:
        r = requests.get(config.KOSIS_URL, params={
            "method": "getList", "apiKey": config.KOSIS_API_KEY, "orgId": "111",
            "tblId": KOSIS_INBOUND_TBL, "itmId": KOSIS_INBOUND_ITM, "objL1": "ALL",
            "objL2": KOSIS_INBOUND_SEX_TOTAL, "objL3": KOSIS_INBOUND_MONTH_TOTAL,
            "prdSe": "Y", "startPrdDe": str(config.HISTORY_BASE_YEAR),
            "endPrdDe": str(_current_quarter()[0]), "format": "json", "jsonVD": "Y"}, timeout=45)
        data = r.json()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, list) or not data:
        return None
    total, by = {}, {}
    for rec in data:
        nm = (rec.get("C1_NM") or "").strip()
        try:
            y = int(rec.get("PRD_DE")); v = float(rec.get("DT") or 0)
        except (TypeError, ValueError):
            continue
        if not nm or v <= 0:
            continue
        if nm == "총계":
            total[y] = v
        elif nm.endswith("주계") or nm in ("기타", "국적미상"):
            continue  # 대륙 집계행 제외
        else:
            by.setdefault(nm, {})[y] = v
    if not total:
        return None
    return {"total": total, "by": by}


def _customs_annual(hs_codes):
    """관세청 무역통계 → 분야별 전국 연간 수출액(2018~). HS 코드 합, 연 단위.

    뷰티(화장품 HS)·패션(의류 HS)·푸드(K-food 가공식품 HS)의 공식 무역통계 기반.
    키 없거나 응답 결측이면 None(→ 검증고정 폴백).
    """
    if not config.DATA_GO_KR_API_KEY or not hs_codes:
        return None
    # 진행 중인 올해는 부분 연도(예: 5개월치)라 1년치와 비교 불가 → 직전 완성연도까지만 사용.
    # (이후 분기는 _idx_q_from_annual이 최신 완성연도 값으로 유지)
    end_year = _current_quarter()[0] - 1
    out, got = {}, False
    for y in range(config.HISTORY_BASE_YEAR, end_year + 1):
        tot = 0.0
        for hs in hs_codes:
            try:
                r = requests.get(config.CUSTOMS_EXPORT_URL, params={
                    "serviceKey": config.DATA_GO_KR_API_KEY,
                    "strtYymm": f"{y}01", "endYymm": f"{y}12", "hsSgn": hs}, timeout=40)
                tot += collectors._parse_customs_expdlr(r.text)
            except Exception:  # noqa: BLE001
                continue
        if tot > 0:
            out[y] = tot
            got = True
    return out if got else None


def interp_annual_to_quarterly(annual: dict, qs: list) -> dict:
    """연간값 → 분기값. 연도 내 다음 연도로 선형 램프, 데이터 밖은 CAGR 외삽."""
    yrs = sorted(annual)
    res = {}
    last = yrs[-1]
    cagr = 1.0
    if len(yrs) >= 2 and annual[yrs[-2]] > 0:
        cagr = (annual[last] / annual[yrs[-2]]) ** (1.0 / (last - yrs[-2]))
    for (y, q) in qs:
        if y in annual and (y + 1) in annual:
            v = annual[y] + (annual[y + 1] - annual[y]) * (q - 1) / 4.0
        elif y in annual:
            slope = (annual[y] - annual[y - 1]) if (y - 1) in annual else 0.0
            v = annual[y] + slope * (q - 1) / 4.0
        elif y > last:
            v = annual[last] * (cagr ** ((y - last) + (q - 1) / 4.0))
        else:
            v = annual[yrs[0]]
        res[(y, q)] = max(v, 0.001)
    return res


# ── 실 백필: KTO 월별 → 분기 ──────────────────────────────────
def _kto_quarterly(qs: list):
    if not config.DATA_GO_KR_API_KEY:
        return None
    monthly = {}
    got = False
    for (y, q) in qs:
        for mo in range((q - 1) * 3 + 1, (q - 1) * 3 + 4):
            ym = f"{y}{mo:02d}"
            try:
                r = requests.get(config.KTO_VISITORS_URL, params={
                    "ServiceKey": config.DATA_GO_KR_API_KEY, "YM": ym, "ED_CD": "E",
                    "numOfRows": 600, "pageNo": 1}, timeout=30)
                tot = sum(int(x) for x in re.findall(r"<num>(\d+)</num>", r.text))
            except Exception:  # noqa: BLE001
                tot = 0
            if tot:
                got = True
            monthly[(y, q)] = monthly.get((y, q), 0) + tot
    if not got:
        return None
    return {k: max(v, 0.001) for k, v in monthly.items()}


def build_history(sample: bool = False):
    base = config.HISTORY_BASE_YEAR
    qs = [q for q in quarters(base) if q <= (2025, 4)]  # 연간비교는 최신 완성연도(2025)까지로 캡
    genres = list(config.GENRE_WEIGHTS)

    # 콘텐츠 4분야(K-pop·K영상·게임·웹툰)의 2018=100 역사 시계열은 검증된 연간 공식통계
    # (REAL_ANNUAL_INDEX)로 결정론적으로 고정한다. 연간 수출 통계는 확정되면 바뀌지 않으므로
    # 실행마다 재호출하지 않는다. KOSIS 콘텐츠산업조사 '수출' 라이브 합산은 방송·만화 항목이
    # 알려진 실적과 크게 어긋나(K영상 라이브 ≈1.0배 vs 검증 ≈2.6배, K웹툰 ≈6.5배 vs 검증 4.9배)
    # 표시 시계열의 안정성·신뢰성을 위해 채택하지 않는다.
    # (아래 관세청·KOSIS-inbound 라이브가 푸드·패션·뷰티·관광을 덮어쓴다.)
    idx, sources = {}, {}
    for g in genres:
        idx[g] = {qq: _real_idx_q(g, qq[0], qq[1]) for qq in qs}
        sources[g] = "stat(verified-annual)"

    # 관광: KOSIS 국적별 외래객 총계(orgId=111, 2018–2024)를 라이브로 받아 2018=100 지수화.
    # 2025는 표에 없어 공표 연간(REAL_ANNUAL_INDEX 2025=122)으로 보강. 실패 시 전부 검증고정 폴백.
    inbound = None if sample else _kosis_inbound_annual()
    if inbound and inbound["total"].get(config.HISTORY_BASE_YEAR):
        tb = dict(inbound["total"])
        base_v = tb[config.HISTORY_BASE_YEAR]
        t_aidx = {y: tb[y] / base_v * 100 for y in tb}
        t_aidx.setdefault(2025, REAL_ANNUAL_INDEX["ktourism"].get(2025, t_aidx[max(t_aidx)]))
        idx["ktourism"] = {qq: _idx_q_from_annual(t_aidx, qq[0], qq[1]) for qq in qs}
        sources["ktourism"] = "kosis-inbound(orgId=111 국적별 2018-2024)+공표2025"
    else:
        sources["ktourism"] = "kto(공표연간 2018-2025 verified-fixed)"

    # 푸드·패션·뷰티는 관세청 무역통계(전국 연간 수출)로 2018=100 지수화 — API 자동화.
    # 푸드=가공식품(라면·김치 등) HS, 패션=의류 HS, 뷰티=화장품 HS. 실패 시 검증고정 유지.
    if not sample:
        for g in ("kfood", "kfashion", "kbeauty"):
            ann = _customs_annual(config.CUSTOMS_HS_CODES.get(g))
            yrs = sorted(y for y in (ann or {}) if y >= config.HISTORY_BASE_YEAR)
            if ann and len(yrs) >= 2 and ann.get(config.HISTORY_BASE_YEAR):
                base_v = ann[config.HISTORY_BASE_YEAR]
                aidx = {y: ann[y] / base_v * 100 for y in ann if ann[y]}
                idx[g] = {qq: _idx_q_from_annual(aidx, qq[0], qq[1]) for qq in qs}
                sources[g] = "customs(real)"

    weights = processor.active_weights()
    composite = {qq: round(sum(weights[g] * idx[g][qq] for g in genres), 1) for qq in qs}

    # 관광 source-market 다각화 지수(2018=100): 국적별 입국자(KOSIS orgId=111)로 유효시장수 ENM=1/HHI.
    # 라이브 국적별 데이터가 있으면 전(全) 국적 기준으로, 없으면 명명 상위시장(TOURISM_SHARES)으로 산출.
    # 중국 집중 완화·시장 다변화 시 상승. 볼륨 지수와 분리된 별도 항목(관광 회복의 질·복원력).
    def _enm_named(shares):
        named = {k: v for k, v in shares.items() if k != "others"}
        tot = sum(named.values()) or 1.0
        fr = [v / tot for v in named.values()]
        hhi = sum(f * f for f in fr) or 1.0
        return 1.0 / hhi

    div_ann = None
    if inbound and inbound["by"]:
        by, tot = inbound["by"], inbound["total"]
        tmp = {}
        for y in sorted(tot):
            names = [c for c in by if by[c].get(y)]
            T = sum(by[c][y] for c in names)
            if T <= 0 or len(names) < 3:
                continue  # 국적별 분해 결측연도(예: 2019)는 건너뜀 → 보간
            hhi = sum((by[c][y] / T) ** 2 for c in names)
            tmp[y] = (1.0 / hhi) if hhi else 0.0
        if 2018 in tmp:
            div_ann = tmp
    if not div_ann:
        div_ann = {y: _enm_named(s) for y, s in TOURISM_SHARES.items()}
    div_base = div_ann.get(base) or div_ann[min(div_ann)]
    div_aidx = {y: div_ann[y] / div_base * 100 for y in div_ann}
    div_series = {qq: _idx_q_from_annual(div_aidx, qq[0], qq[1]) for qq in qs}

    def label(yq):
        return f"{yq[0]}Q{yq[1]}"

    disp = [yq for yq in qs if label(yq) >= config.HISTORY_DISPLAY_START]
    real_n = sum(1 for s in sources.values() if "real" in s or "kosis-inbound" in s)
    return {
        "base_year": base,
        "display_start": config.HISTORY_DISPLAY_START,
        "weight_profile": config.ACTIVE_WEIGHT_PROFILE,
        "sources": sources,
        "real_domains": real_n,
        "note": "2018=100 분기지수(2018–2025 캡). 콘텐츠 4분야(K-pop·K영상·게임·웹툰)=검증 연간 공식통계 고정값(확정 통계라 실행 간 불변), 푸드·패션·뷰티=관세청 무역통계(가공식품·의류·화장품 HS) 라이브, 관광=KOSIS 국적별 외래객(orgId=111, 2018–2024)+2025 공표. 라이브 결측은 검증 통계로 폴백. 분기 선형보간, 최신연도 이후 유지.",
        "notes": {
            "kwebtoon": "K웹툰 L1은 KOSIS '만화 수출' 기준. 콘텐츠산업조사의 만화산업은 출판만화+온라인만화(웹툰)를 포함하며, 2020년부터 웹툰이 매출의 과반·수출을 주도(종이 만화 아님). 2018년 수출 베이스가 작아 증가 배수가 크게 보이는 저(低)베이스 효과 유의. 웹툰 '산업 매출'(2024년 2.29조원, 4.9배)은 내수 포함이라 별도 지표로 해석.",
            "kfood": "K푸드 L1은 관세청 가공식품(라면·김치·소스·조미김 등) 수출 기준. 농식품 총수출(곡물·축산 등 포함)이 아니라 한류 식품 신호에 맞춘 가공식품 위주이므로 총수출보다 증가율이 높을 수 있음.",
            "ktourism": "K관광 L1은 KOSIS 국적·지역별 외국인 입국자(orgId=111, DT_091_111_2009_S005A) 총계. 2018 1,563만→2024 1,697만명(2018=100→약 109), 2025는 공표 연간(약 1,870만, 122)으로 보강. 2020–21 코로나 급감·이후 회복.",
            "tourism_diversification": "방한 외래객의 source-market 다각화 지수(2018=100). KOSIS 국적별 입국자(orgId=111) 전 국적 구성으로 유효시장수 ENM=1/HHI를 산출해 지수화. 값이 높을수록 특정국(특히 중국) 의존이 낮아 시장구성이 다변화·복원력↑. 코로나기(2020–22) 중국 급감으로 분산 최고, 2024 중국(27%) 재집중으로 하락. 관광 '볼륨'과 분리한 '질' 지표.",
            "vintage": "연간비교 2025년까지 캡(2026 부분연도 제외). 콘텐츠(검증 고정값)·관광은 최신 확정연도 이후 동일값 유지.",
        },
        "quarters": [label(q) for q in disp],
        "composite": [composite[q] for q in disp],
        "domains": {g: [idx[g][q] for q in disp] for g in genres},
        "domain_names": {g: config.GENRE_NAMES_KO[g] for g in genres},
        "tourism_diversification": [div_series[q] for q in disp],
        "tourism_shares": TOURISM_SHARES,
    }


def run(sample: bool = False) -> int:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hist = build_history(sample=sample)
    (config.ROOT_DIR / "data" / "history.json").write_text(
        json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[history] {hist['quarters'][0]}~{hist['quarters'][-1]} · 2018=100 · "
          f"실데이터 {hist['real_domains']}/8 · composite {hist['composite'][0]}→{hist['composite'][-1]}")
    for g, s in hist["sources"].items():
        print(f"   {config.GENRE_NAMES_KO[g]:4} {s}")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="강제 샘플(실 백필 생략)")
    return run(sample=ap.parse_args().sample)


if __name__ == "__main__":
    raise SystemExit(main())
