from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "output"

_YT_RAW = os.getenv("YOUTUBE_API_KEY", "")
# 쉼표로 여러 키 등록 가능. 한도(403 quotaExceeded) 시 다음 키로 자동 회전.
YOUTUBE_API_KEYS = [k.strip() for k in _YT_RAW.split(",") if k.strip()]
YOUTUBE_API_KEY = YOUTUBE_API_KEYS[0] if YOUTUBE_API_KEYS else ""
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "").strip()
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
DATA_GO_KR_API_KEY = os.getenv("DATA_GO_KR_API_KEY", "").strip()
KF_API_KEY = os.getenv("KF_API_KEY", DATA_GO_KR_API_KEY).strip()
KF_API_URL = os.getenv("KF_API_URL", "").strip()
KOSIS_API_KEY = os.getenv("KOSIS_API_KEY", "").strip()  # kosis.kr/openapi 발급 키(별도)
YOUTUBE_SEARCH_FALLBACK = os.getenv("YOUTUBE_SEARCH_FALLBACK", "1").strip() not in {"0", "false", "False"}
YOUTUBE_SEARCH_MAX_RESULTS = int(os.getenv("YOUTUBE_SEARCH_MAX_RESULTS", "8"))

# 지수 시간단위: 분기(Quarter) — 프레임워크 기준. 연간 통계와 플랫폼 데이터 갱신주기 조율.
# 개별 수집기는 소스 고유 주기(일/주/월)로 돌고, 지수는 분기 단위로 산출/지수화한다.
REFRESH_CADENCE = "quarterly"
BASE_YEAR = 2018          # 기준연도(=100). 2018Q1~현재 기준으로 Min-Max 정규화.
INDEX_BASE_VALUE = 100

TARGET_COUNTRIES = {
    "US": {"name_ko": "미국", "reddit_terms": ["usa", "america", "united states"]},
    "CN": {"name_ko": "중국", "reddit_terms": ["china", "chinese"]},
    "JP": {"name_ko": "일본", "reddit_terms": ["japan", "japanese"]},
    "VN": {"name_ko": "베트남", "reddit_terms": ["vietnam", "vietnamese"]},
    "TH": {"name_ko": "태국", "reddit_terms": ["thailand", "thai"]},
    "ID": {"name_ko": "인도네시아", "reddit_terms": ["indonesia", "indonesian"]},
    "IN": {"name_ko": "인도", "reddit_terms": ["india", "indian"]},
    "MY": {"name_ko": "말레이시아", "reddit_terms": ["malaysia", "malaysian"]},
    "FR": {"name_ko": "프랑스", "reddit_terms": ["france", "french"]},
    "GB": {"name_ko": "영국", "reddit_terms": ["uk", "britain", "british", "england"]},
    "BR": {"name_ko": "브라질", "reddit_terms": ["brazil", "brazilian"]},
    "AR": {"name_ko": "아르헨티나", "reddit_terms": ["argentina", "argentinian"]},
    "AE": {"name_ko": "UAE", "reddit_terms": ["uae", "dubai", "emirates"]},
    "TR": {"name_ko": "터키", "reddit_terms": ["turkey", "turkish"]},
    "ZA": {"name_ko": "남아공", "reddit_terms": ["south africa", "south african"]},
}

# ── 8개 산업 장르 (2026 개편) ───────────────────────────────────────────
# kpop      : K팝
# kvideo    : K영상 (구 kdrama + kmovie 통합)
# kgame     : K게임
# kwebtoon  : K웹툰
# kfood     : K푸드
# kfashion  : K패션
# kbeauty   : K뷰티(화장품)
# ktourism  : K관광
# 가중치 합 = 1.00 (산업규모 반영). 언제든 이 딕셔너리만 수정하면 지수에 반영됨.
GENRE_WEIGHTS = {
    "kpop": 0.20,
    "kvideo": 0.18,
    "kgame": 0.16,
    "kwebtoon": 0.10,
    "kfood": 0.10,
    "kbeauty": 0.10,
    "kfashion": 0.08,
    "ktourism": 0.08,
}

GENRE_NAMES_KO = {
    "kpop": "K팝",
    "kvideo": "K영상",
    "kgame": "K게임",
    "kwebtoon": "K웹툰",
    "kfood": "K푸드",
    "kbeauty": "K뷰티",
    "kfashion": "K패션",
    "ktourism": "K관광",
}

# ── 프레임워크 3층위(L1/L2/L3) 구조 ────────────────────────────
# L1 경제적 파급(수출·생산유발) / L2 문화적 영향력(차트·플랫폼·수상) / L3 수용자 반응(검색·SNS).
# DSI_i = Σ(층위가중 × 층위_norm). 도메인에 특정 층위 데이터가 없으면 남은 층위로 재정규화.
LAYER_WEIGHTS = {"L1": 0.50, "L2": 0.30, "L3": 0.20}

# L3(수용자) 내부 신호 결합 가중치. Reddit 폐기 → Google Trends로 대체.
L3_SUBWEIGHTS = {"survey": 0.40, "youtube": 0.35, "trends": 0.25}
L2_SIGNAL_WEIGHTS = L3_SUBWEIGHTS  # 하위호환 별칭

# 현재 가동 신호 → 층위 매핑
#   L2: KF 한류현황(국가별 인프라/위상)
#   L3: KOFICE 설문 + YouTube + Google Trends
SIGNAL_LAYER = {"kf": "L2", "survey": "L3", "youtube": "L3", "trends": "L3"}

# Google 차단/제한국 → L3에서 trends 가중을 youtube로 이전 (중국 = 구글 차단)
TRENDS_RESTRICTED = {"CN"}
REDDIT_RESTRICTED = {"CN", "RU"}  # (legacy, reddit 미사용)

# ── Google Trends(pytrends) — L3 수용자 검색 관심도 ────────────
# 분야당 대표 검색어 1개. interest_by_region(COUNTRY)로 분야당 1회 호출에 전 국가 수신.
TRENDS_QUERIES = {
    "kpop": "K-pop", "kvideo": "Korean drama", "kgame": "Korean game",
    "kwebtoon": "webtoon", "kfood": "Korean food", "kbeauty": "Korean skincare",
    "kfashion": "Korean fashion", "ktourism": "Korea travel",
}
# 국가코드 → pytrends interest_by_region 국가명
TRENDS_GEO_NAME = {
    "US": "United States", "CN": "China", "JP": "Japan", "VN": "Vietnam", "TH": "Thailand",
    "ID": "Indonesia", "IN": "India", "MY": "Malaysia", "FR": "France", "GB": "United Kingdom",
    "BR": "Brazil", "AR": "Argentina", "AE": "United Arab Emirates", "TR": "Turkey", "ZA": "South Africa",
}

# ── 2축 도메인 가중치 (w_i = α·수출점유 + β·영향력점유) ─────────
# 현재 활성 w_i는 GENRE_WEIGHTS(산업규모안). 아래는 2축 재산정용 참고치.
WEIGHT_ALPHA = 0.6  # 0.6 경제중심 / 0.4 문화중심
# 도메인 기준 수출액(억$, 2024 추정; 게임·웹툰·패션은 잠정). 2축 가중치 재산정 입력.
DOMAIN_EXPORT_REF = {
    "kpop": 1.8, "kvideo": 13.5, "kgame": 7.0, "kwebtoon": 1.0,
    "kfood": 70.2, "kbeauty": 102.0, "kfashion": 60.0, "ktourism": 5.2,
}

# ── L1 경제 수집: 관세청 품목별 국가별 수출입실적 (data.go.kr 15100475) ──
# DATA_GO_KR_API_KEY 사용. 분야별 HS 코드 수출액(USD)을 국가별로 수집 → L1 경제 층위.
# ── UN Comtrade: 키가 필요 없는 국가별 무역통계 (L1 1차 경로) ──────
# 관세청 API 를 대체한다. 이유가 셋이다.
#   (1) **키가 필요 없다.** data.go.kr 인증키·활용신청·쿼터에 의존하지 않는다
#   (2) **K팝을 덮는다.** 실물 광학매체(HS 852349)로 가중 0.20 도메인의
#       국가별 L1 이 생긴다. 국가별 L1 커버리지가 0.28 → 0.48 로 올라간다
#   (3) **검증됐다.** 기존 준거와 대조한 결과 국가 배분 상관 0.89~1.00,
#       48개 국가-도메인 조합 전부 전년동월 로그차분 상관이 양수다
#
# 주의: HS 8523 을 **4자리로 쓰면 안 된다.** SSD·메모리카드가 대부분이라
# 2023년 최대 파트너 한 곳에만 60억 달러가 잡힌다. 음반은 6자리 852349 다.
COMTRADE_URL = "https://comtradeapi.un.org/public/v1/preview/C/M/HS"
COMTRADE_HS = {
    "kpop":     "852349",                      # 광학 기록 매체(CD·DVD)
    "kfood":    "1902,2005,2103,2008,1212",
    "kfashion": "6101,6102,6109,6203,6204",
    "kbeauty":  "3304,3303,3305,3307",
}
COMTRADE_M49 = {
    842: "US", 392: "JP", 156: "CN", 158: "TW", 704: "VN", 360: "ID", 764: "TH",
    458: "MY", 608: "PH", 76: "BR", 124: "CA", 276: "DE", 250: "FR", 826: "GB",
    356: "IN", 484: "MX", 682: "SA", 784: "AE", 32: "AR", 792: "TR", 710: "ZA",
}
# Comtrade 는 2~9개월 공표 시차가 있다. 고정 오프셋을 쓰면 빈 응답이 오므로
# 최신 월부터 뒤로 훑어 자료가 있는 달을 **한 번만** 찾아 전 도메인에 재사용한다.
COMTRADE_LOOKBACK = 20

CUSTOMS_EXPORT_URL = "http://apis.data.go.kr/1220000/nitemtrade/getNitemtradeList"
# 분야별 HS 헤딩. 라이브 API에서 자릿수 매칭 확인 후 조정 가능(현재 4자리 헤딩 기준).
CUSTOMS_HS_CODES = {
    "kfood": ["1902", "2005", "2103", "2008", "1212"],      # 라면·김치·소스·조미김·건해조
    "kfashion": ["6101", "6102", "6109", "6203", "6204"],   # 의류 대표 헤딩(니트/직물)
    "kbeauty": ["3304", "3303", "3305", "3307"],            # 화장품(스킨/메이크업·향수·헤어 등)
}

# ── 관광 공급 인프라: 한국관광공사 TourAPI KorService2 (data.go.kr 15101578) ──
# 전국 단위 공급 스냅샷(콘텐츠 타입별 총개수). DATA_GO_KR_API_KEY 사용.
TOURAPI_URL = "https://apis.data.go.kr/B551011/KorService2/areaBasedList2"
TOURAPI_CONTENT_TYPES = {
    12: "관광지", 14: "문화시설", 15: "축제공연행사", 25: "여행코스",
    28: "레포츠", 32: "숙박", 38: "쇼핑", 39: "음식점",
}

# ── KOSIS 콘텐츠산업조사 수출액 (orgId=113). 전국·연 단위 → 도메인 L1 컨텍스트 ──
KOSIS_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
KOSIS_ORG_ID = "113"
KOSIS_EXPORT_TBL = "DT_113_STBL_1024776"   # 콘텐츠산업: 수출액 현황 (천달러)
KOSIS_EXPORT_ITM = "13103843981T1"          # 수출액
# 분류값(C1_NM) → 우리 8분야 매핑. 방송+영화 = kvideo(합산).
KOSIS_INDUSTRY_MAP = {"음악": "kpop", "방송": "kvideo", "영화": "kvideo",
                      "게임": "kgame", "만화": "kwebtoon"}

# ── KTO 방한 외래관광객 (한국문화관광연구원 출입국관광통계, data.go.kr 15000297) ──
# 국가별 월별 방한객 수 → ktourism의 per-country L1(관광 수요). NAT_CD 대신 국가명 매칭.
KTO_VISITORS_URL = "http://openapi.tour.go.kr/openapi/service/EdrcntTourismStatsService/getEdrcntTourismStatsList"
KTO_NAME_MAP = {
    "미국": "US", "중국": "CN", "일본": "JP", "베트남": "VN", "태국": "TH",
    "인도네시아": "ID", "인도": "IN", "말레이시아": "MY", "프랑스": "FR", "영국": "GB",
    "브라질": "BR", "아르헨티나": "AR", "아랍에미리트": "AE", "UAE": "AE",
    "터키": "TR", "튀르키예": "TR", "남아프리카공화국": "ZA", "남아공": "ZA",
}

# ── 분기 시계열(history) ────────────────────────────────────────
HISTORY_BASE_YEAR = 2018       # 지수화 기준연도(=100)
HISTORY_DISPLAY_START = "2021Q1"  # 대시보드 표시 시작

# ── 가중치 프로파일 (여러 옵션을 동시에 열어둠) ──────────────────
# 활성 프로파일이 KWCI 산출에 쓰이고, 모든 프로파일의 비교가 결과 JSON에 함께 기록된다.
#   industry          : 산업규모안 (현 GENRE_WEIGHTS)
#   equal             : 8분야 동일(0.125)
#   two_axis_economic : w=α·수출점유+(1-α)·영향력, α=0.6 (경제중심)
#   two_axis_cultural : 동 공식 α=0.4 (문화중심)
# 변경은 이 한 줄만 바꾸면 된다.
ACTIVE_WEIGHT_PROFILE = "two_axis_cultural"

# 글로벌 종합 지수 집계: "mean"(단순평균) 또는 "pop_weighted"(인구가중평균).
# 두 값 모두 산출되며, 이 설정이 대표 헤드라인 값(global_index)을 결정한다.
GLOBAL_INDEX_METHOD = "mean"

# 인구가중평균용 대략 인구(백만, 2025 추정). 청중 규모 프록시.
COUNTRY_POPULATION = {
    "US": 340, "CN": 1410, "JP": 124, "VN": 99, "TH": 72,
    "ID": 277, "IN": 1430, "MY": 34, "FR": 66, "GB": 68,
    "BR": 216, "AR": 46, "AE": 10, "TR": 86, "ZA": 60,
}

GENRE_KEYWORDS = {
    "kpop": [
        "kpop", "k-pop", "bts", "blackpink", "stray kids", "newjeans", "twice",
        "seventeen", "aespa", "le sserafim", "ive", "txt", "enhypen", "nct",
    ],
    "kvideo": [
        "kdrama", "k-drama", "korean drama", "korean series", "netflix korea",
        "korean film", "korean movie", "k-movie", "parasite", "squid game",
        "train to busan", "kingdom", "the glory",
    ],
    "kgame": [
        "korean game", "lost ark", "maplestory", "pubg", "dave the diver",
        "lies of p", "stellar blade", "black desert", "the first descendant",
        "nexon", "krafton", "ncsoft", "shift up",
    ],
    "kwebtoon": [
        "webtoon", "manhwa", "korean webtoon", "solo leveling", "tower of god",
        "lookism", "naver webtoon", "lezhin", "true beauty", "noblesse",
    ],
    "kfood": [
        "kfood", "k-food", "korean food", "kimchi", "ramyeon", "korean bbq",
        "tteokbokki", "bibimbap", "soju", "korean fried chicken", "mukbang",
    ],
    "kbeauty": [
        "kbeauty", "k-beauty", "korean skincare", "korean makeup",
        "korean cosmetics", "glass skin", "cosrx", "laneige", "innisfree", "anua",
    ],
    "kfashion": [
        "korean fashion", "k-fashion", "korean style", "seoul fashion",
        "korean streetwear", "musinsa", "hanbok", "korean outfit",
    ],
    "ktourism": [
        "korea travel", "visit korea", "seoul travel", "korea tourism", "jeju",
        "busan travel", "korea vlog", "travel korea", "incheon",
    ],
}

YOUTUBE_SEARCH_QUERIES = {
    "kpop": "K-pop official music video",
    "kvideo": "Korean drama movie official trailer",
    "kgame": "Korean game official gameplay trailer",
    "kwebtoon": "webtoon manhwa motion comic",
    "kfood": "Korean food recipe mukbang",
    "kbeauty": "Korean skincare kbeauty routine review",
    "kfashion": "Korean fashion lookbook outfit",
    "ktourism": "Korea travel vlog Seoul",
}

GENRE_SUBREDDITS = {
    "kpop": ["kpop", "bangtan", "blackpink", "twice", "seventeen"],
    "kvideo": ["kdrama", "KDRAMA", "koreanvariety", "movies"],
    "kgame": ["lostarkgame", "Maplestory", "PUBATTLEGROUNDS", "gaming"],
    "kwebtoon": ["webtoons", "manhwa", "sololeveling", "TowerofGod"],
    "kfood": ["KoreanFood", "food", "Cooking"],
    "kbeauty": ["AsianBeauty", "kbeauty", "SkincareAddiction"],
    "kfashion": ["koreanfashionadvice", "streetwear", "femalefashionadvice"],
    "ktourism": ["korea", "KoreaTravel", "travel"],
}

# KOFICE 해외한류실태조사 baseline (연 1회 설문 기준선, 8장르 개편판).
# 각 값은 0~100 관심/호감 지표. kf = KF 한류 인프라 baseline.
SAMPLE_SURVEY = {
    "US": {"kpop": 72, "kvideo": 55, "kgame": 64, "kwebtoon": 52, "kfood": 42, "kbeauty": 55, "kfashion": 40, "ktourism": 38, "kf": 78},
    "CN": {"kpop": 64, "kvideo": 58, "kgame": 55, "kwebtoon": 56, "kfood": 49, "kbeauty": 73, "kfashion": 53, "ktourism": 42, "kf": 60},
    "JP": {"kpop": 69, "kvideo": 56, "kgame": 58, "kwebtoon": 54, "kfood": 51, "kbeauty": 59, "kfashion": 42, "ktourism": 50, "kf": 72},
    "VN": {"kpop": 83, "kvideo": 72, "kgame": 70, "kwebtoon": 68, "kfood": 57, "kbeauty": 67, "kfashion": 48, "ktourism": 47, "kf": 64},
    "TH": {"kpop": 88, "kvideo": 72, "kgame": 66, "kwebtoon": 64, "kfood": 48, "kbeauty": 62, "kfashion": 45, "ktourism": 37, "kf": 65},
    "ID": {"kpop": 86, "kvideo": 68, "kgame": 69, "kwebtoon": 66, "kfood": 54, "kbeauty": 61, "kfashion": 44, "ktourism": 41, "kf": 69},
    "IN": {"kpop": 54, "kvideo": 42, "kgame": 54, "kwebtoon": 48, "kfood": 32, "kbeauty": 42, "kfashion": 30, "ktourism": 32, "kf": 50},
    "MY": {"kpop": 79, "kvideo": 65, "kgame": 60, "kwebtoon": 58, "kfood": 52, "kbeauty": 65, "kfashion": 47, "ktourism": 39, "kf": 58},
    "FR": {"kpop": 60, "kvideo": 44, "kgame": 56, "kwebtoon": 50, "kfood": 39, "kbeauty": 48, "kfashion": 35, "ktourism": 44, "kf": 46},
    "GB": {"kpop": 58, "kvideo": 43, "kgame": 55, "kwebtoon": 47, "kfood": 36, "kbeauty": 45, "kfashion": 32, "ktourism": 40, "kf": 44},
    "BR": {"kpop": 76, "kvideo": 52, "kgame": 62, "kwebtoon": 54, "kfood": 40, "kbeauty": 51, "kfashion": 37, "ktourism": 36, "kf": 55},
    "AR": {"kpop": 68, "kvideo": 45, "kgame": 58, "kwebtoon": 50, "kfood": 36, "kbeauty": 44, "kfashion": 32, "ktourism": 33, "kf": 43},
    "AE": {"kpop": 63, "kvideo": 50, "kgame": 52, "kwebtoon": 44, "kfood": 43, "kbeauty": 54, "kfashion": 39, "ktourism": 35, "kf": 40},
    "TR": {"kpop": 66, "kvideo": 49, "kgame": 58, "kwebtoon": 50, "kfood": 35, "kbeauty": 46, "kfashion": 33, "ktourism": 32, "kf": 42},
    "ZA": {"kpop": 52, "kvideo": 36, "kgame": 50, "kwebtoon": 42, "kfood": 30, "kbeauty": 36, "kfashion": 26, "ktourism": 28, "kf": 34},
}
