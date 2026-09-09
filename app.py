import io
import math
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# FUTBOL QUANT ENGINE
# V3 - DISCIPLINED VALUE / RISK / BANKROLL ENGINE
# ============================================================

st.set_page_config(
    page_title="Futbol Quant Engine",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ FUTBOL QUANT ENGINE")
st.caption(
    "2 istatistik dosyası + 1 oran dosyası | "
    "Model → MODEL LOCK → Fair Odds → EV → Monte Carlo → Risk"
)

# ------------------------------------------------------------
# GENEL AYARLAR
# ------------------------------------------------------------

DEFAULT_SIMULATIONS = 100_000


# ------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ------------------------------------------------------------

def normalize_name(x):
    if pd.isna(x):
        return ""

    x = str(x).strip().lower()

    replacements = {
        "ı": "i",
        "İ": "i",
        "ş": "s",
        "Ş": "s",
        "ğ": "g",
        "Ğ": "g",
        "ü": "u",
        "Ü": "u",
        "ö": "o",
        "Ö": "o",
        "ç": "c",
        "Ç": "c",
    }

    for a, b in replacements.items():
        x = x.replace(a, b)

    x = re.sub(r"[^a-z0-9]+", "_", x)
    return x.strip("_")


def find_column(df, candidates):
    normalized = {
        normalize_name(c): c
        for c in df.columns
    }

    # Önce tam eşleşme
    for candidate in candidates:
        n = normalize_name(candidate)
        if n in normalized:
            return normalized[n]

    # Sonra içinde geçiyorsa
    for candidate in candidates:
        n = normalize_name(candidate)
        for col_norm, original in normalized.items():
            if n in col_norm or col_norm in n:
                return original

    return None


def read_uploaded_file(uploaded):
    """
    CSV / XLSX / XLS okumaya çalışır.
    """
    name = uploaded.name.lower()

    if name.endswith(".csv"):
        raw = uploaded.read()

        for encoding in ["utf-8-sig", "utf-8", "cp1254", "latin1"]:
            try:
                text = raw.decode(encoding)
                return pd.read_csv(io.StringIO(text))
            except Exception:
                pass

        raise ValueError("CSV dosyası okunamadı.")

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)

    raise ValueError(
        f"Desteklenmeyen dosya: {uploaded.name}. "
        "CSV, XLSX veya XLS kullan."
    )


def clean_numeric(series):
    if series is None:
        return pd.Series(dtype=float)

    s = (
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.extract(r"(-?\d+(?:\.\d+)?)")[0]
    )

    return pd.to_numeric(s, errors="coerce")


def extract_goals_from_score(value):
    if pd.isna(value):
        return None, None

    text = str(value).strip()

    # 2-1, 2:1, 2 – 1 gibi skorları yakalar
    m = re.search(r"(\d+)\s*[-:]\s*(\d+)", text)

    if not m:
        return None, None

    return int(m.group(1)), int(m.group(2))


def prepare_match_data(df):
    """
    Farklı sütun isimlerini mümkün olduğunca otomatik tanır.
    """

    result = df.copy()

    home_col = find_column(
        result,
        [
            "home_team",
            "home",
            "ev_sahibi",
            "ev sahibi",
            "home team",
            "team_home",
            "host"
        ]
    )

    away_col = find_column(
        result,
        [
            "away_team",
            "away",
            "deplasman",
            "deplasman takimi",
            "away team",
            "team_away",
            "visitor"
        ]
    )

    hg_col = find_column(
        result,
        [
            "home_goals",
            "home goals",
            "home_score",
            "ev_gol",
            "ev golleri",
            "hg",
            "fthg"
        ]
    )

    ag_col = find_column(
        result,
        [
            "away_goals",
            "away goals",
            "away_score",
            "dep_gol",
            "deplasman golleri",
            "ag",
            "ftag"
        ]
    )

    score_col = find_column(
        result,
        [
            "score",
            "result",
            "ft_result",
            "full_time",
            "full time",
            "mac_skoru",
            "skor"
        ]
    )

    # Skor tek sütundaysa ayır
    if (hg_col is None or ag_col is None) and score_col is not None:
        parsed = result[score_col].apply(extract_goals_from_score)

        if hg_col is None:
            result["_home_goals_auto"] = parsed.apply(
                lambda x: x[0] if x else np.nan
            )
            hg_col = "_home_goals_auto"

        if ag_col is None:
            result["_away_goals_auto"] = parsed.apply(
                lambda x: x[1] if x else np.nan
            )
            ag_col = "_away_goals_auto"

    if home_col is None or away_col is None:
        return None, {
            "error": (
                "Ev sahibi/deplasman takım sütunları bulunamadı."
            )
        }

    if hg_col is None or ag_col is None:
        return None, {
            "error": (
                "Ev sahibi/deplasman gol sütunları bulunamadı. "
                "Model için maç sonucu/gol bilgisi gerekiyor."
            )
        }

    result["_home_team"] = result[home_col].astype(str).str.strip()
    result["_away_team"] = result[away_col].astype(str).str.strip()

    result["_home_goals"] = clean_numeric(result[hg_col])
    result["_away_goals"] = clean_numeric(result[ag_col])

    result = result.dropna(
        subset=["_home_team", "_away_team", "_home_goals", "_away_goals"]
    )

    result = result[
        (result["_home_team"] != "") &
        (result["_away_team"] != "")
    ]

    return result, {
        "home": home_col,
        "away": away_col,
        "home_goals": hg_col,
        "away_goals": ag_col,
    }


def combine_stat_files(df1, df2):
    frames = []

    for df in [df1, df2]:
        prepared, info = prepare_match_data(df)

        if prepared is not None and len(prepared) > 0:
            frames.append(prepared)

    if not frames:
        return pd.DataFrame(), "İki istatistik dosyasından maç geçmişi çıkarılamadı."

    combined = pd.concat(frames, ignore_index=True)

    # Aynı maç iki dosyada da varsa mümkün olduğunca temizle
    combined["_match_key"] = (
        combined["_home_team"].map(normalize_name)
        + "_"
        + combined["_away_team"].map(normalize_name)
        + "_"
        + combined["_home_goals"].astype(str)
        + "_"
        + combined["_away_goals"].astype(str)
    )

    combined = combined.drop_duplicates("_match_key")

    return combined, None


# ------------------------------------------------------------
# TAKIM MODELİ
# ------------------------------------------------------------

def team_strengths(matches, team):
    """
    Basit ve şeffaf hücum/savunma modeli.

    Lambda doğrudan oranlardan üretilmez.
    Sadece geçmiş gol performansı ve lig ortalamalarından oluşturulur.
    """

    if matches.empty:
        return None

    league_home_avg = matches["_home_goals"].mean()
    league_away_avg = matches["_away_goals"].mean()

    if not np.isfinite(league_home_avg) or league_home_avg <= 0:
        league_home_avg = 1.35

    if not np.isfinite(league_away_avg) or league_away_avg <= 0:
        league_away_avg = 1.10

    home_for = matches.loc[
        matches["_home_team"] == team,
        "_home_goals"
    ]

    home_against = matches.loc[
        matches["_home_team"] == team,
        "_away_goals"
    ]

    away_for = matches.loc[
        matches["_away_team"] == team,
        "_away_goals"
    ]

    away_against = matches.loc[
        matches["_away_team"] == team,
        "_home_goals"
    ]

    # Takımın ev performansı
    hf = home_for.mean() if len(home_for) else league_home_avg
    ha = home_against.mean() if len(home_against) else league_away_avg

    # Takımın deplasman performansı
    af = away_for.mean() if len(away_for) else league_away_avg
    aa = away_against.mean() if len(away_against) else league_home_avg

    return {
        "home_attack": hf / league_home_avg,
        "home_defense": ha / league_away_avg,
        "away_attack": af / league_away_avg,
        "away_defense": aa / league_home_avg,
        "league_home_avg": league_home_avg,
        "league_away_avg": league_away_avg,
        "sample_home": len(home_for),
        "sample_away": len(away_for),
    }


def calculate_lambdas(matches, home_team, away_team):
    """
    Oranlardan bağımsız ilk model.
    """

    league_home_avg = matches["_home_goals"].mean()
    league_away_avg = matches["_away_goals"].mean()

    home = team_strengths(matches, home_team)
    away = team_strengths(matches, away_team)

    if home is None or away is None:
        return None

    # Hücum x Savunma
    raw_home = (
        league_home_avg
        * home["home_attack"]
        * away["away_defense"]
    )

    raw_away = (
        league_away_avg
        * away["away_attack"]
        * home["home_defense"]
    )

    # Aşırı değerleri sınırlama
    lambda_home = float(np.clip(raw_home, 0.15, 4.50))
    lambda_away = float(np.clip(raw_away, 0.15, 4.00))

    return {
        "home": lambda_home,
        "away": lambda_away,
        "home_sample": home["sample_home"],
        "away_sample": away["sample_away"],
        "league_home_avg": league_home_avg,
        "league_away_avg": league_away_avg,
    }


# ------------------------------------------------------------
# POISSON
# ------------------------------------------------------------

def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0

    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def score_matrix(lambda_home, lambda_away, max_goals=8):
    matrix = np.zeros(
        (max_goals + 1, max_goals + 1),
        dtype=float
    )

    for h in range(max_goals + 1):
        ph = poisson_pmf(h, lambda_home)

        for a in range(max_goals + 1):
            pa = poisson_pmf(a, lambda_away)
            matrix[h, a] = ph * pa

    total = matrix.sum()

    if total > 0:
        matrix /= total

    return matrix


def market_probabilities(matrix):
    home = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away = np.triu(matrix, 1).sum()

    return {
        "1": float(home),
        "X": float(draw),
        "2": float(away),
    }


# ------------------------------------------------------------
# MONTE CARLO
# ------------------------------------------------------------

def monte_carlo(lambda_home, lambda_away, simulations=100_000, seed=42):
    rng = np.random.default_rng(seed)

    home_goals = rng.poisson(
        lambda_home,
        simulations
    )

    away_goals = rng.poisson(
        lambda_away,
        simulations
    )

    home_prob = np.mean(home_goals > away_goals)
    draw_prob = np.mean(home_goals == away_goals)
    away_prob = np.mean(home_goals < away_goals)

    over25 = np.mean(
        (home_goals + away_goals) >= 3
    )

    under25 = 1 - over25

    btts = np.mean(
        (home_goals > 0) &
        (away_goals > 0)
    )

    return {
        "1": float(home_prob),
        "X": float(draw_prob),
        "2": float(away_prob),
        "Over 2.5": float(over25),
        "Under 2.5": float(under25),
        "BTTS Yes": float(btts),
        "BTTS No": float(1 - btts),
        "home_goals": float(home_goals.mean()),
        "away_goals": float(away_goals.mean()),
    }


# ------------------------------------------------------------
# ORAN DOSYASI
# ------------------------------------------------------------

def detect_odds_columns(df):
    home = find_column(
        df,
        [
            "home_odds",
            "home odds",
            "ev oran",
            "ev sahibi oran",
            "1",
            "home"
        ]
    )

    draw = find_column(
        df,
        [
            "draw_odds",
            "draw odds",
            "beraberlik oran",
            "x",
            "draw"
        ]
    )

    away = find_column(
        df,
        [
            "away_odds",
            "away odds",
            "deplasman oran",
            "2",
            "away"
        ]
    )

    return home, draw, away


def parse_odds(value):
    try:
        if pd.isna(value):
            return np.nan

        text = str(value).replace(",", ".")

        m = re.search(r"\d+(?:\.\d+)?", text)

        if not m:
            return np.nan

        value = float(m.group())

        if value <= 1.0:
            return np.nan

        return value

    except Exception:
        return np.nan


def no_vig_probabilities(home_odds, draw_odds, away_odds):
    inv = np.array([
        1 / home_odds,
        1 / draw_odds,
        1 / away_odds
    ])

    total = inv.sum()

    return {
        "1": inv[0] / total,
        "X": inv[1] / total,
        "2": inv[2] / total,
    }


# ------------------------------------------------------------
# EV
# ------------------------------------------------------------

def fair_odds(prob):
    if prob <= 0:
        return np.inf

    return 1 / prob


def expected_value(prob, odds):
    return prob * odds - 1


def kelly_fraction(prob, odds):
    b = odds - 1

    if b <= 0:
        return 0

    q = 1 - prob

    k = ((b * prob) - q) / b

    return max(0.0, k)


# ------------------------------------------------------------
# ANALİZ
# ------------------------------------------------------------

def analyze_market(model_prob, market_prob, odds):
    rows = []

    for market in ["1", "X", "2"]:
        p = model_prob[market]
        mp = market_prob[market]
        odd = odds[market]

        fair = fair_odds(p)
        ev = expected_value(p, odd)
        kelly = kelly_fraction(p, odd)

        # Model-market farkı
        edge = p - mp

        # Quarter Kelly
        qk = kelly * 0.25

        # Maksimum %2 sermaye sınırı
        stake_pct = min(qk, 0.02)

        if ev >= 0.05 and edge >= 0.03:
            decision = "BET"
        else:
            decision = "NO BET"

        rows.append({
            "Piyasa": market,
            "Model Olasılığı": p,
            "Piyasa Olasılığı": mp,
            "Fair Odds": fair,
            "Oran": odd,
            "EV": ev,
            "Edge": edge,
            "Kelly": kelly,
            "Quarter Kelly": qk,
            "Önerilen Bankroll %": stake_pct,
            "Karar": decision
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# ARAYÜZ
# ------------------------------------------------------------

st.markdown("---")

st.subheader("📂 1. Dosyaları yükle")

col1, col2 = st.columns(2)

with col1:
    stat_file_1 = st.file_uploader(
        "📊 İstatistik Dosyası 1",
        type=["csv", "xlsx", "xls"],
        key="stat1"
    )

with col2:
    stat_file_2 = st.file_uploader(
        "📊 İstatistik Dosyası 2",
        type=["csv", "xlsx", "xls"],
        key="stat2"
    )

odds_file = st.file_uploader(
    "💰 Oran Dosyası",
    type=["csv", "xlsx", "xls"],
    key="odds"
)


if not stat_file_1 or not stat_file_2 or not odds_file:
    st.info(
        "Başlamak için 2 istatistik dosyasını ve 1 oran dosyasını yükle."
    )

    st.markdown("""
### Sistem nasıl çalışacak?

**1️⃣ İstatistik dosyaları**

Geçmiş maç verilerinden takım performansı çıkarılır.

**2️⃣ MODEL LOCK**

Oranlara bakmadan Home/Away λ üretilir.

**3️⃣ Poisson**

1-X-2 olasılıkları hesaplanır.

**4️⃣ Oran analizi**

Oranların implied probability değeri ve vig temizlenmiş piyasa olasılığı hesaplanır.

**5️⃣ Monte Carlo**

100.000 simülasyonla model tekrar kontrol edilir.

**6️⃣ Value**

Fair Odds, EV ve Edge hesaplanır.

**7️⃣ Risk**

Quarter Kelly uygulanır ve maksimum bankroll riski sınırlandırılır.

**8️⃣ Sonuç**

Avantaj yeterli değilse:

### ❌ NO BET

Avantaj yeterliyse:

### ✅ BET
""")

    st.stop()


# ------------------------------------------------------------
# DOSYALARI OKU
# ------------------------------------------------------------

try:
    df1 = read_uploaded_file(stat_file_1)
    df2 = read_uploaded_file(stat_file_2)
    odds_df = read_uploaded_file(odds_file)

except Exception as e:
    st.error(f"Dosya okunamadı: {e}")
    st.stop()


# ------------------------------------------------------------
# MAÇ VERİLERİNİ BİRLEŞTİR
# ------------------------------------------------------------

matches, combine_error = combine_stat_files(df1, df2)

if combine_error:
    st.error(combine_error)

    st.write("Dosya 1 sütunları:")
    st.write(list(df1.columns))

    st.write("Dosya 2 sütunları:")
    st.write(list(df2.columns))

    st.stop()


st.success(
    f"✅ {len(matches):,} kullanılabilir maç kaydı bulundu."
)


# ------------------------------------------------------------
# TAKIMLAR
# ------------------------------------------------------------

teams = sorted(
    set(matches["_home_team"].unique())
    | set(matches["_away_team"].unique())
)

st.markdown("---")
st.subheader("⚽ 2. Analiz edilecek maç")

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox(
        "Ev Sahibi",
        teams
    )

with col2:
    away_options = [
        x for x in teams
        if x != home_team
    ]

    away_team = st.selectbox(
        "Deplasman",
        away_options
    )


if home_team == away_team:
    st.error("Ev sahibi ve deplasman aynı takım olamaz.")
    st.stop()


# ------------------------------------------------------------
# MODEL LOCK
# ------------------------------------------------------------

st.markdown("---")
st.subheader("🔒 MODEL LOCK")

model = calculate_lambdas(
    matches,
    home_team,
    away_team
)

if model is None:
    st.error("Model oluşturulamadı.")
    st.stop()


lambda_home = model["home"]
lambda_away = model["away"]

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "λ Home",
        f"{lambda_home:.3f}"
    )

with c2:
    st.metric(
        "λ Away",
        f"{lambda_away:.3f}"
    )

with c3:
    st.metric(
        "Toplam λ",
        f"{lambda_home + lambda_away:.3f}"
    )

st.success(
    "🔒 MODEL LOCK tamamlandı. "
    "Bu aşamadaki λ değerleri oranlardan bağımsız üretilmiştir."
)


# ------------------------------------------------------------
# POISSON
# ------------------------------------------------------------

st.markdown("---")
st.subheader("📊 3. Poisson Modeli")

matrix = score_matrix(
    lambda_home,
    lambda_away,
    max_goals=8
)

model_probs = market_probabilities(matrix)

p1, px, p2 = st.columns(3)

with p1:
    st.metric(
        "1",
        f"{model_probs['1'] * 100:.2f}%"
    )

with px:
    st.metric(
        "X",
        f"{model_probs['X'] * 100:.2f}%"
    )

with p2:
    st.metric(
        "2",
        f"{model_probs['2'] * 100:.2f}%"
    )


# En olası skorlar
score_rows = []

for h in range(matrix.shape[0]):
    for a in range(matrix.shape[1]):
        score_rows.append({
            "Skor": f"{h}-{a}",
            "Olasılık": matrix[h, a]
        })

score_df = pd.DataFrame(score_rows)

score_df = score_df.sort_values(
    "Olasılık",
    ascending=False
).head(10)

score_df["Olasılık"] = (
    score_df["Olasılık"] * 100
).round(2)

st.write("### 🎯 En olası skorlar")
st.dataframe(
    score_df,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# MONTE CARLO
# ------------------------------------------------------------

st.markdown("---")
st.subheader("🎲 4. Monte Carlo")

simulations = st.number_input(
    "Simülasyon sayısı",
    min_value=100_000,
    max_value=2_000_000,
    value=100_000,
    step=100_000
)

if st.button(
    "🎲 Monte Carlo Simülasyonunu Çalıştır",
    use_container_width=True
):

    mc = monte_carlo(
        lambda_home,
        lambda_away,
        int(simulations),
        seed=42
    )

    m1, mx, m2 = st.columns(3)

    with m1:
        st.metric(
            "Monte Carlo 1",
            f"{mc['1'] * 100:.2f}%"
        )

    with mx:
        st.metric(
            "Monte Carlo X",
            f"{mc['X'] * 100:.2f}%"
        )

    with m2:
        st.metric(
            "Monte Carlo 2",
            f"{mc['2'] * 100:.2f}%"
        )

    st.write(
        f"Over 2.5: **{mc['Over 2.5'] * 100:.2f}%**"
    )

    st.write(
        f"BTTS Yes: **{mc['BTTS Yes'] * 100:.2f}%**"
    )

    st.write(
        f"Ortalama skor: "
        f"**{mc['home_goals']:.2f} - {mc['away_goals']:.2f}**"
    )


# ------------------------------------------------------------
# ORANLAR
# ------------------------------------------------------------

st.markdown("---")
st.subheader("💰 5. Oran Dosyası")

odds_home_col, odds_draw_col, odds_away_col = detect_odds_columns(
    odds_df
)

st.write("Bulunan oran sütunları:")

st.write({
    "1": odds_home_col,
    "X": odds_draw_col,
    "2": odds_away_col
})


# Otomatik bulamazsa kullanıcı seçsin
all_odds_columns = list(odds_df.columns)

if odds_home_col is None:
    odds_home_col = st.selectbox(
        "Ev sahibi oran sütunu",
        all_odds_columns,
        key="oh"
    )

if odds_draw_col is None:
    odds_draw_col = st.selectbox(
        "Beraberlik oran sütunu",
        all_odds_columns,
        key="ox"
    )

if odds_away_col is None:
    odds_away_col = st.selectbox(
        "Deplasman oran sütunu",
        all_odds_columns,
        key="oa"
    )


odds_home = parse_odds(
    odds_df[odds_home_col].iloc[0]
)

odds_draw = parse_odds(
    odds_df[odds_draw_col].iloc[0]
)

odds_away = parse_odds(
    odds_df[odds_away_col].iloc[0]
)


if not all(
    np.isfinite(x)
    for x in [odds_home, odds_draw, odds_away]
):
    st.error(
        "İlk satırda geçerli 1-X-2 oranları bulunamadı."
    )
    st.stop()


st.write(
    f"**Oranlar:** "
    f"1 = {odds_home:.2f} | "
    f"X = {odds_draw:.2f} | "
    f"2 = {odds_away:.2f}"
)


# ------------------------------------------------------------
# NO VIG
# ------------------------------------------------------------

market_probs = no_vig_probabilities(
    odds_home,
    odds_draw,
    odds_away
)

st.write(
    "### 📉 Vig temizlenmiş piyasa olasılıkları"
)

nv1, nvx, nv2 = st.columns(3)

with nv1:
    st.metric(
        "Piyasa 1",
        f"{market_probs['1'] * 100:.2f}%"
    )

with nvx:
    st.metric(
        "Piyasa X",
        f"{market_probs['X'] * 100:.2f}%"
    )

with nv2:
    st.metric(
        "Piyasa 2",
        f"{market_probs['2'] * 100:.2f}%"
    )


# ------------------------------------------------------------
# VALUE / EV
# ------------------------------------------------------------

st.markdown("---")
st.subheader("📈 6. VALUE / EV ANALİZİ")

odds = {
    "1": odds_home,
    "X": odds_draw,
    "2": odds_away
}

analysis_df = analyze_market(
    model_probs,
    market_probs,
    odds
)

display_df = analysis_df.copy()

for col in [
    "Model Olasılığı",
    "Piyasa Olasılığı",
    "EV",
    "Edge",
    "Kelly",
    "Quarter Kelly",
    "Önerilen Bankroll %"
]:
    display_df[col] = (
        display_df[col] * 100
    ).round(2)

display_df["Fair Odds"] = display_df["Fair Odds"].round(3)
display_df["Oran"] = display_df["Oran"].round(2)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# KARAR MOTORU
# ------------------------------------------------------------

st.markdown("---")
st.subheader("🧠 7. QUANT KARAR MOTORU")

bets = analysis_df[
    analysis_df["Karar"] == "BET"
].copy()


if bets.empty:

    st.error(
        "❌ NO BET — Model tarafından yeterli istatistiksel değer bulunamadı."
    )

else:

    best = bets.sort_values(
        "EV",
        ascending=False
    ).iloc[0]

    st.success(
        f"✅ VALUE BULUNDU → {best['Piyasa']}"
    )

    bc1, bc2, bc3, bc4 = st.columns(4)

    with bc1:
        st.metric(
            "Piyasa",
            best["Piyasa"]
        )

    with bc2:
        st.metric(
            "Oran",
            f"{best['Oran']:.2f}"
        )

    with bc3:
        st.metric(
            "EV",
            f"{best['EV'] * 100:.2f}%"
        )

    with bc4:
        st.metric(
            "Fair Odds",
            f"{best['Fair Odds']:.2f}"
        )

    st.write(
        f"Model olasılığı: "
        f"**{best['Model Olasılığı'] * 100:.2f}%**"
    )

    st.write(
        f"Piyasa olasılığı: "
        f"**{best['Piyasa Olasılığı'] * 100:.2f}%**"
    )

    st.write(
        f"Edge: "
        f"**{best['Edge'] * 100:.2f}%**"
    )


# ------------------------------------------------------------
# BANKROLL
# ------------------------------------------------------------

st.markdown("---")
st.subheader("💵 8. BANKROLL / QUARTER KELLY")

bankroll = st.number_input(
    "Bankroll (TL)",
    min_value=0.0,
    value=1000.0,
    step=100.0
)

if not bets.empty and bankroll > 0:

    best = bets.sort_values(
        "EV",
        ascending=False
    ).iloc[0]

    stake = bankroll * min(
        best["Quarter Kelly"],
        0.02
    )

    st.metric(
        "Önerilen maksimum bahis",
        f"{stake:.2f} TL"
    )

    st.caption(
        "Risk kontrolü: Quarter Kelly ve maksimum %2 bankroll sınırı."
    )


# ------------------------------------------------------------
# VERİ KALİTESİ
# ------------------------------------------------------------

st.markdown("---")
st.subheader("🔍 9. VERİ KALİTESİ")

q1, q2, q3 = st.columns(3)

with q1:
    st.metric(
        "Toplam maç",
        f"{len(matches):,}"
    )

with q2:
    st.metric(
        f"{home_team} örneklem",
        f"{len(matches[(matches['_home_team'] == home_team) | (matches['_away_team'] == home_team)]):,}"
    )

with q3:
    st.metric(
        f"{away_team} örneklem",
        f"{len(matches[(matches['_home_team'] == away_team) | (matches['_away_team'] == away_team)]):,}"
    )


st.markdown("---")

st.caption(
    "Futbol Quant Engine — İstatistiksel analiz aracıdır. "
    "Tahmin veya bahis sonucu garanti etmez."
)
