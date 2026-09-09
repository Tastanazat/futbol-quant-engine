import streamlit as st
from supabase import create_client, Client
from datetime import datetime, timezone
import pandas as pd
import math
import json
import statistics


# =========================================================
# FUTBOL QUANT ENGINE
# MAÇ ÖNCESİ QUANT ENGINE
# =========================================================

st.set_page_config(
    page_title="Futbol Quant Engine",
    page_icon="⚽",
    layout="wide"
)

MODEL_VERSION = "FQE-PRE-1.0"


# =========================================================
# SUPABASE
# =========================================================

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

    supabase: Client = create_client(
        SUPABASE_URL,
        SUPABASE_KEY
    )

    db_ok = True

except Exception as e:
    db_ok = False
    supabase = None

    st.error(
        "Supabase bağlantısı kurulamadı.\n\n"
        "Streamlit Secrets içine SUPABASE_URL ve "
        "SUPABASE_KEY eklenmelidir."
    )


# =========================================================
# GENEL YARDIMCILAR
# =========================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def pct(value):
    return f"{safe_float(value) * 100:.2f}%"


def fair_odd(probability):
    if probability is None or probability <= 0:
        return None

    return round(1.0 / probability, 3)


def poisson_probability(lmbda, goals):
    """
    P(X=k)
    """

    lmbda = max(0.0001, float(lmbda))
    goals = int(goals)

    return (
        math.exp(-lmbda)
        * (lmbda ** goals)
        / math.factorial(goals)
    )


def poisson_distribution(lmbda, max_goals=10):
    probs = [
        poisson_probability(lmbda, i)
        for i in range(max_goals + 1)
    ]

    total = sum(probs)

    if total > 0:
        probs = [x / total for x in probs]

    return probs


# =========================================================
# DATABASE HELPERS
# =========================================================

def fetch_table(table, select="*"):
    if not db_ok:
        return []

    try:
        response = (
            supabase
            .table(table)
            .select(select)
            .execute()
        )

        return response.data or []

    except Exception as e:
        st.error(f"{table} okunamadı: {e}")
        return []


def insert_row(table, data):
    if not db_ok:
        return None

    try:
        response = (
            supabase
            .table(table)
            .insert(data)
            .execute()
        )

        if response.data:
            return response.data[0]

        return None

    except Exception as e:
        st.error(f"{table} kayıt hatası: {e}")
        return None


def update_row(table, record_id, data):
    if not db_ok:
        return False

    try:
        (
            supabase
            .table(table)
            .update(data)
            .eq("id", record_id)
            .execute()
        )

        return True

    except Exception as e:
        st.error(f"{table} güncelleme hatası: {e}")
        return False


def log_audit(table_name, record_id, action,
              old_data=None, new_data=None):

    if not db_ok:
        return

    try:
        data = {
            "table_name": table_name,
            "record_id": record_id,
            "action": action,
            "old_data": old_data,
            "new_data": new_data
        }

        (
            supabase
            .table("data_audit_log")
            .insert(data)
            .execute()
        )

    except Exception:
        pass


# =========================================================
# VERİYİ ÇEK
# =========================================================

def get_matches():
    return fetch_table(
        "matches",
        "*"
    )


def get_statistics():
    return fetch_table(
        "match_statistics",
        "*"
    )


def get_odds():
    return fetch_table(
        "odds",
        "*"
    )


def get_predictions():
    return fetch_table(
        "predictions",
        "*"
    )


def get_prediction_results():
    return fetch_table(
        "prediction_results",
        "*"
    )


# =========================================================
# DATAFRAME OLUŞTUR
# =========================================================

def prepare_history():

    matches = pd.DataFrame(get_matches())
    stats = pd.DataFrame(get_statistics())

    if matches.empty:
        return pd.DataFrame()

    matches["match_date"] = pd.to_datetime(
        matches["match_date"],
        errors="coerce"
    )

    if not stats.empty:

        stats["match_id"] = stats["match_id"].astype(str)
        matches["id"] = matches["id"].astype(str)

        df = matches.merge(
            stats,
            left_on="id",
            right_on="match_id",
            how="left"
        )

    else:
        df = matches.copy()

    return df


# =========================================================
# TAMAMLANMIŞ MAÇLAR
# =========================================================

def completed_history():

    df = prepare_history()

    if df.empty:
        return df

    df = df[
        df["home_score"].notna()
        & df["away_score"].notna()
    ].copy()

    if "status" in df.columns:

        completed_status = [
            "completed",
            "finished",
            "settled",
            "played"
        ]

        status_mask = (
            df["status"]
            .astype(str)
            .str.lower()
            .isin(completed_status)
        )

        # Skor varsa status farklı olsa bile maçı koru.
        df = df[
            status_mask
            | (
                df["home_score"].notna()
                & df["away_score"].notna()
            )
        ]

    return df


# =========================================================
# TAKIM FORM VERİSİ
# =========================================================

def team_match_rows(team, history):

    if history.empty:
        return pd.DataFrame()

    home = history[
        history["home_team"].astype(str).str.lower()
        == team.lower()
    ].copy()

    away = history[
        history["away_team"].astype(str).str.lower()
        == team.lower()
    ].copy()

    return pd.concat(
        [home, away],
        ignore_index=True
    )


# =========================================================
# SON MAÇLAR
# =========================================================

def recent_team_matches(team, history, limit=10):

    df = team_match_rows(team, history)

    if df.empty:
        return df

    if "match_date" in df.columns:
        df = df.sort_values(
            "match_date",
            ascending=False
        )

    return df.head(limit)


# =========================================================
# TAKIM METRİKLERİ
# =========================================================

def team_metrics(team, history, limit=10):

    df = recent_team_matches(
        team,
        history,
        limit
    )

    if df.empty:
        return {
            "matches": 0,
            "gf": None,
            "ga": None,
            "xgf": None,
            "xga": None,
            "shots": None,
            "shots_on_target": None,
            "corners": None
        }

    rows = []

    for _, r in df.iterrows():

        is_home = (
            str(r["home_team"]).lower()
            == team.lower()
        )

        if is_home:

            gf = safe_float(r.get("home_score"))
            ga = safe_float(r.get("away_score"))

            xgf = safe_float(r.get("home_xg"), None)
            xga = safe_float(r.get("away_xg"), None)

            shots = safe_float(
                r.get("home_shots"),
                None
            )

            sot = safe_float(
                r.get("home_shots_on_target"),
                None
            )

            corners = safe_float(
                r.get("home_corners"),
                None
            )

        else:

            gf = safe_float(r.get("away_score"))
            ga = safe_float(r.get("home_score"))

            xgf = safe_float(r.get("away_xg"), None)
            xga = safe_float(r.get("home_xg"), None)

            shots = safe_float(
                r.get("away_shots"),
                None
            )

            sot = safe_float(
                r.get("away_shots_on_target"),
                None
            )

            corners = safe_float(
                r.get("away_corners"),
                None
            )

        rows.append({
            "gf": gf,
            "ga": ga,
            "xgf": xgf,
            "xga": xga,
            "shots": shots,
            "sot": sot,
            "corners": corners
        })

    result = {
        "matches": len(rows),
    }

    for key in [
        "gf",
        "ga",
        "xgf",
        "xga",
        "shots",
        "sot",
        "corners"
    ]:

        values = [
            x[key]
            for x in rows
            if x[key] is not None
        ]

        result[key] = (
            sum(values) / len(values)
            if values
            else None
        )

    return result


# =========================================================
# EV
# =========================================================

def expected_value(probability, odds):

    probability = safe_float(probability)
    odds = safe_float(odds)

    if probability <= 0 or odds <= 1:
        return None

    return probability * odds - 1


# =========================================================
# QUARTER KELLY
# =========================================================

def quarter_kelly(probability, odds,
                  bankroll,
                  max_fraction=0.02):

    probability = safe_float(probability)
    odds = safe_float(odds)
    bankroll = safe_float(bankroll)

    if (
        probability <= 0
        or odds <= 1
        or bankroll <= 0
    ):
        return 0.0

    b = odds - 1

    q = 1 - probability

    full_kelly = (
        (b * probability - q) / b
    )

    if full_kelly <= 0:
        return 0.0

    stake_fraction = min(
        full_kelly * 0.25,
        max_fraction
    )

    return round(
        bankroll * stake_fraction,
        2
    )


# =========================================================
# λ MODEL
# =========================================================

def calculate_lambdas(
    home_team,
    away_team,
    history,
    sample=10
):

    home_recent = recent_team_matches(
        home_team,
        history,
        sample
    )

    away_recent = recent_team_matches(
        away_team,
        history,
        sample
    )

    home_metrics = team_metrics(
        home_team,
        history,
        sample
    )

    away_metrics = team_metrics(
        away_team,
        history,
        sample
    )

    if (
        home_metrics["matches"] == 0
        or away_metrics["matches"] == 0
    ):
        return None

    # -----------------------------------------------------
    # TEMEL GOL MODELİ
    # -----------------------------------------------------

    home_attack = (
        home_metrics["xgf"]
        if home_metrics["xgf"] is not None
        else home_metrics["gf"]
    )

    home_defense = (
        home_metrics["xga"]
        if home_metrics["xga"] is not None
        else home_metrics["ga"]
    )

    away_attack = (
        away_metrics["xgf"]
        if away_metrics["xgf"] is not None
        else away_metrics["gf"]
    )

    away_defense = (
        away_metrics["xga"]
        if away_metrics["xga"] is not None
        else away_metrics["ga"]
    )

    # Attack + opponent defence blend
    raw_home = (
        0.60 * home_attack
        + 0.40 * away_defense
    )

    raw_away = (
        0.60 * away_attack
        + 0.40 * home_defense
    )

    # -----------------------------------------------------
    # SONUÇ GOLLERİ İLE xG'Yİ BLEND ET
    # -----------------------------------------------------

    if (
        home_metrics["xgf"] is not None
        and home_metrics["gf"] is not None
    ):

        home_goal_signal = (
            0.70 * home_metrics["xgf"]
            + 0.30 * home_metrics["gf"]
        )

        raw_home = (
            0.70 * raw_home
            + 0.30 * home_goal_signal
        )

    if (
        away_metrics["xgf"] is not None
        and away_metrics["gf"] is not None
    ):

        away_goal_signal = (
            0.70 * away_metrics["xgf"]
            + 0.30 * away_metrics["gf"]
        )

        raw_away = (
            0.70 * raw_away
            + 0.30 * away_goal_signal
        )

    # -----------------------------------------------------
    # EV SAHİBİ AVANTAJI
    # -----------------------------------------------------

    raw_home *= 1.08
    raw_away *= 0.94

    # -----------------------------------------------------
    # AŞIRI λ'YI SINIRLA
    # -----------------------------------------------------

    lambda_home = max(
        0.15,
        min(raw_home, 4.50)
    )

    lambda_away = max(
        0.15,
        min(raw_away, 4.50)
    )

    return {
        "lambda_home": round(lambda_home, 4),
        "lambda_away": round(lambda_away, 4),
        "home_metrics": home_metrics,
        "away_metrics": away_metrics
    }


# =========================================================
# POISSON MAÇ OLASILIKLARI
# =========================================================

def calculate_match_probabilities(
    lambda_home,
    lambda_away,
    max_goals=10
):

    home_dist = poisson_distribution(
        lambda_home,
        max_goals
    )

    away_dist = poisson_distribution(
        lambda_away,
        max_goals
    )

    home_win = 0
    draw = 0
    away_win = 0

    over_15 = 0
    over_25 = 0
    over_35 = 0

    btts_yes = 0

    score_matrix = []

    for h in range(max_goals + 1):

        row = []

        for a in range(max_goals + 1):

            p = (
                home_dist[h]
                * away_dist[a]
            )

            row.append(p)

            if h > a:
                home_win += p

            elif h == a:
                draw += p

            else:
                away_win += p

            total = h + a

            if total >= 2:
                over_15 += p

            if total >= 3:
                over_25 += p

            if total >= 4:
                over_35 += p

            if h >= 1 and a >= 1:
                btts_yes += p

        score_matrix.append(row)

    # normalize 1X2
    total_1x2 = (
        home_win
        + draw
        + away_win
    )

    if total_1x2 > 0:

        home_win /= total_1x2
        draw /= total_1x2
        away_win /= total_1x2

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "btts_yes": btts_yes,
        "btts_no": 1 - btts_yes,
        "over_15": over_15,
        "under_15": 1 - over_15,
        "over_25": over_25,
        "under_25": 1 - over_25,
        "over_35": over_35,
        "under_35": 1 - over_35,
        "matrix": score_matrix
    }


# =========================================================
# TOP SCORE
# =========================================================

def top_scores(
    lambda_home,
    lambda_away,
    count=10
):

    scores = []

    for h in range(0, 8):

        for a in range(0, 8):

            p = (
                poisson_probability(lambda_home, h)
                * poisson_probability(lambda_away, a)
            )

            scores.append({
                "Skor": f"{h}-{a}",
                "Olasılık": p
            })

    scores.sort(
        key=lambda x: x["Olasılık"],
        reverse=True
    )

    return scores[:count]


# =========================================================
# MARKETLER
# =========================================================

def build_markets(prob):

    return [
        {
            "market": "1X2",
            "selection": "1",
            "probability": prob["home_win"]
        },
        {
            "market": "1X2",
            "selection": "X",
            "probability": prob["draw"]
        },
        {
            "market": "1X2",
            "selection": "2",
            "probability": prob["away_win"]
        },
        {
            "market": "KG",
            "selection": "VAR",
            "probability": prob["btts_yes"]
        },
        {
            "market": "KG",
            "selection": "YOK",
            "probability": prob["btts_no"]
        },
        {
            "market": "ÜST/ALT 1.5",
            "selection": "ÜST 1.5",
            "probability": prob["over_15"]
        },
        {
            "market": "ÜST/ALT 1.5",
            "selection": "ALT 1.5",
            "probability": prob["under_15"]
        },
        {
            "market": "ÜST/ALT 2.5",
            "selection": "ÜST 2.5",
            "probability": prob["over_25"]
        },
        {
            "market": "ÜST/ALT 2.5",
            "selection": "ALT 2.5",
            "probability": prob["under_25"]
        },
        {
            "market": "ÜST/ALT 3.5",
            "selection": "ÜST 3.5",
            "probability": prob["over_35"]
        },
        {
            "market": "ÜST/ALT 3.5",
            "selection": "ALT 3.5",
            "probability": prob["under_35"]
        }
    ]


# =========================================================
# ORANLARI MARKETLERE EŞLEŞTİR
# =========================================================

def odds_for_match(match_id):

    rows = get_odds()

    result = []

    for r in rows:

        if str(r.get("match_id")) == str(match_id):

            result.append(r)

    return result


# =========================================================
# BANKROLL
# =========================================================

def get_current_bankroll(default_bankroll=10000):

    history = fetch_table(
        "bankroll_history",
        "*"
    )

    if not history:
        return default_bankroll

    df = pd.DataFrame(history)

    if df.empty:
        return default_bankroll

    if "created_at" in df.columns:

        df["created_at"] = pd.to_datetime(
            df["created_at"],
            errors="coerce"
        )

        df = df.sort_values(
            "created_at"
        )

    last = df.iloc[-1]

    value = last.get("bankroll_after")

    if value is None:
        return default_bankroll

    return safe_float(
        value,
        default_bankroll
    )


# =========================================================
# PREDICTION KAYDET
# =========================================================

def save_prediction(
    match_id,
    market,
    selection,
    probability,
    fair_odds,
    bookmaker_odds,
    ev,
    confidence,
    decision,
    stake,
    bankroll_before
):

    data = {
        "match_id": match_id,
        "model_version": MODEL_VERSION,
        "market": market,
        "selection": selection,
        "predicted_probability": round(
            probability,
            6
        ),
        "fair_odds": fair_odds,
        "bookmaker_odds": bookmaker_odds,
        "expected_value": (
            round(ev, 6)
            if ev is not None
            else None
        ),
        "confidence": round(
            confidence,
            4
        ),
        "decision": decision,
        "stake": round(stake, 2),
        "bankroll_before": round(
            bankroll_before,
            2
        )
    }

    row = insert_row(
        "predictions",
        data
    )

    if row:

        log_audit(
            "predictions",
            row.get("id"),
            "INSERT",
            None,
            data
        )

    return row


# =========================================================
# BAŞLIK
# =========================================================

st.title("⚽ FUTBOL QUANT ENGINE")

st.caption(
    "MAÇ ÖNCESİ • MODEL LOCK • POISSON • FAIR ODDS • EV • RİSK • KALİBRASYON"
)


if not db_ok:
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Motor Ayarları")

sample_size = st.sidebar.slider(
    "Form örneklemi",
    min_value=4,
    max_value=20,
    value=10
)

min_ev = st.sidebar.slider(
    "Minimum EV",
    min_value=0.00,
    max_value=0.20,
    value=0.05,
    step=0.01
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0.50,
    max_value=0.95,
    value=0.60,
    step=0.01
)

default_bankroll = st.sidebar.number_input(
    "Başlangıç Kasa",
    min_value=0.0,
    value=10000.0,
    step=100.0
)


# =========================================================
# MENÜ
# =========================================================

menu = st.sidebar.radio(
    "MENÜ",
    [
        "🏠 Dashboard",
        "➕ Maç Ekle",
        "📊 Maç Öncesi Analiz",
        "💰 Oran Ekle",
        "🏁 Sonuç Gir",
        "📚 Tahmin Geçmişi",
        "📈 Performans",
        "🧠 Kalibrasyon"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if menu == "🏠 Dashboard":

    st.subheader("🏠 Quant Engine Dashboard")

    history = completed_history()

    predictions = pd.DataFrame(
        get_predictions()
    )

    results = pd.DataFrame(
        get_prediction_results()
    )

    bankroll = get_current_bankroll(
        default_bankroll
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "💰 Kasa",
        f"{bankroll:.2f} TL"
    )

    c2.metric(
        "📚 Tamamlanan Maç",
        len(history)
    )

    c3.metric(
        "🧠 Tahmin",
        len(predictions)
    )

    c4.metric(
        "🏁 Sonuç",
        len(results)
    )

    st.divider()

    if not predictions.empty:

        st.subheader(
            "Son Tahminler"
        )

        cols = [
            "market",
            "selection",
            "predicted_probability",
            "fair_odds",
            "bookmaker_odds",
            "expected_value",
            "confidence",
            "decision",
            "stake"
        ]

        available = [
            x for x in cols
            if x in predictions.columns
        ]

        st.dataframe(
            predictions[
                available
            ].tail(20),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Henüz tahmin oluşturulmadı."
        )


# =========================================================
# MAÇ EKLE
# =========================================================

elif menu == "➕ Maç Ekle":

    st.subheader(
        "➕ Maç Ekle"
    )

    with st.form("match_form"):

        col1, col2 = st.columns(2)

        with col1:

            league = st.text_input(
                "Lig",
                placeholder="Süper Lig"
            )

            home_team = st.text_input(
                "Ev Sahibi"
            )

        with col2:

            match_date = st.datetime_input(
                "Maç Tarihi"
            )

            away_team = st.text_input(
                "Deplasman"
            )

        submit = st.form_submit_button(
            "💾 MAÇI KAYDET",
            type="primary"
        )

    if submit:

        if not home_team or not away_team:

            st.warning(
                "Ev sahibi ve deplasman girilmelidir."
            )

        else:

            data = {
                "match_date": match_date.isoformat(),
                "league": league,
                "home_team": home_team.strip(),
                "away_team": away_team.strip(),
                "status": "pending"
            }

            row = insert_row(
                "matches",
                data
            )

            if row:

                log_audit(
                    "matches",
                    row.get("id"),
                    "INSERT",
                    None,
                    data
                )

                st.success(
                    "✅ Maç oluşturuldu."
                )

                st.code(
                    str(row.get("id"))
                )


# =========================================================
# ORAN EKLE
# =========================================================

elif menu == "💰 Oran Ekle":

    st.subheader(
        "💰 Maç Oranı Ekle"
    )

    matches = get_matches()

    if not matches:

        st.info(
            "Önce maç oluşturmalısın."
        )

    else:

        match_options = {
            f"{m['home_team']} - {m['away_team']} | "
            f"{m.get('league', '')} | "
            f"{str(m.get('match_date', ''))[:16]}":
                m
            for m in matches
        }

        selected_text = st.selectbox(
            "Maç",
            list(match_options.keys())
        )

        match = match_options[
            selected_text
        ]

        with st.form("odds_form"):

            bookmaker = st.text_input(
                "Bookmaker",
                value="Pinnacle"
            )

            market = st.selectbox(
                "Market",
                [
                    "1X2",
                    "KG",
                    "ÜST/ALT 1.5",
                    "ÜST/ALT 2.5",
                    "ÜST/ALT 3.5"
                ]
            )

            selection_options = {
                "1X2": ["1", "X", "2"],
                "KG": ["VAR", "YOK"],
                "ÜST/ALT 1.5": [
                    "ÜST 1.5",
                    "ALT 1.5"
                ],
                "ÜST/ALT 2.5": [
                    "ÜST 2.5",
                    "ALT 2.5"
                ],
                "ÜST/ALT 3.5": [
                    "ÜST 3.5",
                    "ALT 3.5"
                ]
            }

            selection = st.selectbox(
                "Seçim",
                selection_options[
                    market
                ]
            )

            odds_value = st.number_input(
                "Oran",
                min_value=1.01,
                value=1.85,
                step=0.01
            )

            submit = st.form_submit_button(
                "💾 ORANI KAYDET",
                type="primary"
            )

        if submit:

            data = {
                "match_id": match["id"],
                "bookmaker": bookmaker,
                "market": market,
                "selection": selection,
                "odds": float(odds_value)
            }

            row = insert_row(
                "odds",
                data
            )

            if row:

                log_audit(
                    "odds",
                    row.get("id"),
                    "INSERT",
                    None,
                    data
                )

                st.success(
                    "✅ Oran kaydedildi."
                )


# =========================================================
# MAÇ ÖNCESİ ANALİZ
# =========================================================

elif menu == "📊 Maç Öncesi Analiz":

    st.subheader(
        "📊 MAÇ ÖNCESİ QUANT ANALİZİ"
    )

    st.warning(
        "MODEL LOCK: Oranlar karar aşamasından önce "
        "model λ hesaplamasına dahil edilmez."
    )

    matches = get_matches()

    if not matches:

        st.info(
            "Analiz için önce maç ekle."
        )

    else:

        match_options = {
            f"{m['home_team']} - {m['away_team']} | "
            f"{m.get('league', '')}":
                m
            for m in matches
            if str(m.get("status", "pending")).lower()
            == "pending"
        }

        if not match_options:

            st.info(
                "Analiz edilecek pending maç bulunamadı."
            )

        else:

            selected_text = st.selectbox(
                "Analiz edilecek maç",
                list(match_options.keys())
            )

            match = match_options[
                selected_text
            ]

            history = completed_history()

            if history.empty:

                st.error(
                    "Model için tamamlanmış geçmiş maç "
                    "verisi gerekiyor."
                )

            else:

                result = calculate_lambdas(
                    match["home_team"],
                    match["away_team"],
                    history,
                    sample_size
                )

                if result is None:

                    st.error(
                        "Bu iki takım için yeterli geçmiş "
                        "verisi bulunamadı."
                    )

                else:

                    lambda_home = result[
                        "lambda_home"
                    ]

                    lambda_away = result[
                        "lambda_away"
                    ]

                    # =====================================
                    # MODEL LOCK
                    # =====================================

                    prob = calculate_match_probabilities(
                        lambda_home,
                        lambda_away
                    )

                    st.success(
                        "🔒 MODEL LOCK TAMAMLANDI"
                    )

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "HOME λ",
                        f"{lambda_home:.3f}"
                    )

                    c2.metric(
                        "AWAY λ",
                        f"{lambda_away:.3f}"
                    )

                    c3.metric(
                        "Beklenen Toplam Gol",
                        f"{lambda_home + lambda_away:.3f}"
                    )

                    st.divider()

                    # =====================================
                    # TAKIM VERİSİ
                    # =====================================

                    st.subheader(
                        "📊 Takım Form Özeti"
                    )

                    hm = result[
                        "home_metrics"
                    ]

                    am = result[
                        "away_metrics"
                    ]

                    form_df = pd.DataFrame([
                        {
                            "Takım": match["home_team"],
                            "Maç": hm["matches"],
                            "GF": hm["gf"],
                            "GA": hm["ga"],
                            "xGF": hm["xgf"],
                            "xGA": hm["xga"],
                            "Şut": hm["shots"],
                            "İsabetli Şut": hm["sot"],
                            "Korner": hm["corners"]
                        },
                        {
                            "Takım": match["away_team"],
                            "Maç": am["matches"],
                            "GF": am["gf"],
                            "GA": am["ga"],
                            "xGF": am["xgf"],
                            "xGA": am["xga"],
                            "Şut": am["shots"],
                            "İsabetli Şut": am["sot"],
                            "Korner": am["corners"]
                        }
                    ])

                    st.dataframe(
                        form_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # =====================================
                    # 1X2
                    # =====================================

                    st.subheader(
                        "🎯 MODEL OLASILIKLARI"
                    )

                    p1, px, p2 = st.columns(3)

                    p1.metric(
                        "1",
                        pct(prob["home_win"]),
                        f"Fair {fair_odd(prob['home_win'])}"
                    )

                    px.metric(
                        "X",
                        pct(prob["draw"]),
                        f"Fair {fair_odd(prob['draw'])}"
                    )

                    p2.metric(
                        "2",
                        pct(prob["away_win"]),
                        f"Fair {fair_odd(prob['away_win'])}"
                    )

                    c1, c2 = st.columns(2)

                    with c1:

                        st.metric(
                            "KG VAR",
                            pct(prob["btts_yes"]),
                            f"Fair {fair_odd(prob['btts_yes'])}"
                        )

                        st.metric(
                            "KG YOK",
                            pct(prob["btts_no"]),
                            f"Fair {fair_odd(prob['btts_no'])}"
                        )

                    with c2:

                        st.metric(
                            "ÜST 2.5",
                            pct(prob["over_25"]),
                            f"Fair {fair_odd(prob['over_25'])}"
                        )

                        st.metric(
                            "ALT 2.5",
                            pct(prob["under_25"]),
                            f"Fair {fair_odd(prob['under_25'])}"
                        )

                    # =====================================
                    # SKORLAR
                    # =====================================

                    st.subheader(
                        "🔢 En Olası Skorlar"
                    )

                    score_data = top_scores(
                        lambda_home,
                        lambda_away,
                        10
                    )

                    score_df = pd.DataFrame(
                        score_data
                    )

                    score_df["Olasılık"] = (
                        score_df["Olasılık"]
                        * 100
                    ).round(2)

                    score_df.rename(
                        columns={
                            "Olasılık":
                            "Olasılık %"
                        },
                        inplace=True
                    )

                    st.dataframe(
                        score_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # =====================================
                    # ORANLAR
                    # =====================================

                    st.divider()

                    st.subheader(
                        "💰 MODEL vs ORAN"
                    )

                    odds_rows = odds_for_match(
                        match["id"]
                    )

                    if not odds_rows:

                        st.info(
                            "Bu maça henüz oran girilmemiş."
                        )

                    else:

                        markets = build_markets(
                            prob
                        )

                        odds_lookup = {}

                        for o in odds_rows:

                            key = (
                                str(o.get("market")),
                                str(o.get("selection"))
                            )

                            odds_lookup[key] = (
                                safe_float(
                                    o.get("odds")
                                )
                            )

                        analysis_rows = []

                        for m in markets:

                            key = (
                                m["market"],
                                m["selection"]
                            )

                            bookmaker_odds = (
                                odds_lookup.get(key)
                            )

                            probability = (
                                m["probability"]
                            )

                            fodd = fair_odd(
                                probability
                            )

                            ev = None

                            if bookmaker_odds:

                                ev = expected_value(
                                    probability,
                                    bookmaker_odds
                                )

                            # Confidence:
                            # probability + data availability
                            home_n = hm["matches"]
                            away_n = am["matches"]

                            sample_factor = min(
                                1.0,
                                (
                                    home_n
                                    + away_n
                                ) / (
                                    sample_size * 2
                                )
                            )

                            confidence = (
                                0.70
                                * probability
                                + 0.30
                                * sample_factor
                            )

                            decision = "NO BET"

                            if (
                                ev is not None
                                and ev >= min_ev
                                and confidence >= min_confidence
                            ):
                                decision = "VALUE"

                            analysis_rows.append({
                                "Market":
                                    m["market"],
                                "Seçim":
                                    m["selection"],
                                "Model %":
                                    probability * 100,
                                "Fair Odds":
                                    fodd,
                                "Bookmaker":
                                    bookmaker_odds,
                                "EV %":
                                    (
                                        ev * 100
                                        if ev is not None
                                        else None
                                    ),
                                "Confidence %":
                                    confidence * 100,
                                "Karar":
                                    decision
                            })

                        analysis_df = pd.DataFrame(
                            analysis_rows
                        )

                        for col in [
                            "Model %",
                            "EV %",
                            "Confidence %"
                        ]:

                            if col in analysis_df.columns:

                                analysis_df[col] = (
                                    analysis_df[col]
                                    .round(2)
                                )

                        st.dataframe(
                            analysis_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        # =================================
                        # EN İYİ VALUE
                        # =================================

                        valid_values = [
                            x for x in analysis_rows
                            if x["Karar"] == "VALUE"
                        ]

                        if not valid_values:

                            st.warning(
                                "⚠️ Belirlenen kriterlere "
                                "uyan VALUE bulunamadı → NO BET."
                            )

                        else:

                            best = max(
                                valid_values,
                                key=lambda x: safe_float(
                                    x["EV %"]
                                )
                            )

                            st.success(
                                f"🎯 VALUE: "
                                f"{best['Market']} "
                                f"{best['Seçim']} | "
                                f"EV %{best['EV %']:.2f}"
                            )

                            bankroll = (
                                get_current_bankroll(
                                    default_bankroll
                                )
                            )

                            best_probability = (
                                best["Model %"]
                                / 100
                            )

                            best_odds = safe_float(
                                best["Bookmaker"]
                            )

                            stake = quarter_kelly(
                                best_probability,
                                best_odds,
                                bankroll
                            )

                            st.metric(
                                "Önerilen Quarter Kelly",
                                f"{stake:.2f} TL"
                            )

                            # =================================
                            # TAHMİNİ KAYDET
                            # =================================

                            if st.button(
                                "💾 EN İYİ VALUE TAHMİNİNİ KAYDET",
                                type="primary"
                            ):

                                # tekrar kayıtları önlemek için
                                existing = get_predictions()

                                already = False

                                for p in existing:

                                    if (
                                        str(
                                            p.get("match_id")
                                        )
                                        == str(match["id"])
                                        and
                                        p.get("market")
                                        == best["Market"]
                                        and
                                        p.get("selection")
                                        == best["Seçim"]
                                        and
                                        p.get("model_version")
                                        == MODEL_VERSION
                                    ):

                                        already = True

                                if already:

                                    st.warning(
                                        "Bu model tahmini "
                                        "zaten kayıtlı."
                                    )

                                else:

                                    confidence_value = (
                                        best["Confidence %"]
                                        / 100
                                    )

                                    ev_decimal = (
                                        best["EV %"]
                                        / 100
                                    )

                                    row = save_prediction(
                                        match_id=match["id"],
                                        market=best["Market"],
                                        selection=best["Seçim"],
                                        probability=best_probability,
                                        fair_odds=best["Fair Odds"],
                                        bookmaker_odds=best_odds,
                                        ev=ev_decimal,
                                        confidence=confidence_value,
                                        decision="BET",
                                        stake=stake,
                                        bankroll_before=bankroll
                                    )

                                    if row:

                                        st.success(
                                            "✅ Tahmin predictions "
                                            "tablosuna kaydedildi."
                                        )


# =========================================================
# SONUÇ GİR
# =========================================================

elif menu == "🏁 Sonuç Gir":

    st.subheader(
        "🏁 Tahmin Sonucu"
    )

    predictions = get_predictions()

    results = get_prediction_results()

    settled_ids = {
        str(x.get("prediction_id"))
        for x in results
    }

    pending = [
        x for x in predictions
        if str(x.get("id"))
        not in settled_ids
    ]

    if not pending:

        st.info(
            "Sonuç bekleyen tahmin yok."
        )

    else:

        options = {
            f"{x.get('market')} "
            f"{x.get('selection')} | "
            f"Oran {x.get('bookmaker_odds')} | "
            f"Stake {x.get('stake')}":
                x
            for x in pending
        }

        selected_text = st.selectbox(
            "Tahmin seç",
            list(options.keys())
        )

        prediction = options[
            selected_text
        ]

        st.write(
            f"**Market:** {prediction.get('market')}"
        )

        st.write(
            f"**Seçim:** {prediction.get('selection')}"
        )

        st.write(
            f"**Model olasılığı:** "
            f"{safe_float(prediction.get('predicted_probability')) * 100:.2f}%"
        )

        result = st.radio(
            "Sonuç",
            [
                "KAZANDI",
                "KAYBETTİ",
                "İADE"
            ],
            horizontal=True
        )

        if st.button(
            "🏁 SONUCU KAYDET",
            type="primary"
        ):

            stake = safe_float(
                prediction.get("stake")
            )

            odds_value = safe_float(
                prediction.get("bookmaker_odds")
            )

            bankroll_before = safe_float(
                prediction.get("bankroll_before")
            )

            if result == "KAZANDI":

                profit_loss = (
                    stake
                    * (
                        odds_value - 1
                    )
                )

            elif result == "KAYBETTİ":

                profit_loss = -stake

            else:

                profit_loss = 0.0

            bankroll_after = (
                bankroll_before
                + profit_loss
            )

            result_data = {
                "prediction_id":
                    prediction["id"],
                "result":
                    result,
                "profit_loss":
                    round(
                        profit_loss,
                        2
                    ),
                "settled_odds":
                    odds_value,
                "settled_at":
                    utc_now(),
                "manually_corrected":
                    False
            }

            result_row = insert_row(
                "prediction_results",
                result_data
            )

            if result_row:

                log_audit(
                    "prediction_results",
                    result_row.get("id"),
                    "INSERT",
                    None,
                    result_data
                )

                # -----------------------------
                # BANKROLL
                # -----------------------------

                bankroll_data = {
                    "prediction_id":
                        prediction["id"],
                    "bankroll_before":
                        bankroll_before,
                    "stake":
                        stake,
                    "profit_loss":
                        round(
                            profit_loss,
                            2
                        ),
                    "bankroll_after":
                        round(
                            bankroll_after,
                            2
                        )
                }

                bank_row = insert_row(
                    "bankroll_history",
                    bankroll_data
                )

                if bank_row:

                    log_audit(
                        "bankroll_history",
                        bank_row.get("id"),
                        "INSERT",
                        None,
                        bankroll_data
                    )

                st.success(
                    f"✅ {result} | "
                    f"Kâr/Zarar: "
                    f"{profit_loss:.2f} TL | "
                    f"Yeni Kasa: "
                    f"{bankroll_after:.2f} TL"
                )

                st.rerun()


# =========================================================
# TAHMİN GEÇMİŞİ
# =========================================================

elif menu == "📚 Tahmin Geçmişi":

    st.subheader(
        "📚 Tahmin Geçmişi"
    )

    predictions = pd.DataFrame(
        get_predictions()
    )

    if predictions.empty:

        st.info(
            "Henüz tahmin yok."
        )

    else:

        st.dataframe(
            predictions.sort_values(
                "created_at",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# PERFORMANS
# =========================================================

elif menu == "📈 Performans":

    st.subheader(
        "📈 Model Performansı"
    )

    predictions = pd.DataFrame(
        get_predictions()
    )

    results = pd.DataFrame(
        get_prediction_results()
    )

    if predictions.empty or results.empty:

        st.info(
            "Performans için tahmin ve sonuç "
            "verisi gerekiyor."
        )

    else:

        df = predictions.merge(
            results[
                [
                    "prediction_id",
                    "result",
                    "profit_loss"
                ]
            ],
            left_on="id",
            right_on="prediction_id",
            how="inner"
        )

        if df.empty:

            st.info(
                "Henüz sonuçlanmış tahmin yok."
            )

        else:

            won = len(
                df[
                    df["result"]
                    == "KAZANDI"
                ]
            )

            lost = len(
                df[
                    df["result"]
                    == "KAYBETTİ"
                ]
            )

            total = won + lost

            total_profit = df[
                "profit_loss"
            ].fillna(0).sum()

            total_stake = df[
                "stake"
            ].fillna(0).sum()

            win_rate = (
                won / total * 100
                if total > 0
                else 0
            )

            roi = (
                total_profit
                / total_stake
                * 100
                if total_stake > 0
                else 0
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Kazandı",
                won
            )

            c2.metric(
                "Kaybetti",
                lost
            )

            c3.metric(
                "Win Rate",
                f"{win_rate:.2f}%"
            )

            c4.metric(
                "ROI",
                f"{roi:.2f}%"
            )

            st.metric(
                "Toplam Kâr/Zarar",
                f"{total_profit:.2f} TL"
            )

            st.divider()

            st.subheader(
                "Tahmin Sonuçları"
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# KALİBRASYON
# =========================================================

elif menu == "🧠 Kalibrasyon":

    st.subheader(
        "🧠 MODEL KALİBRASYONU"
    )

    predictions = pd.DataFrame(
        get_predictions()
    )

    results = pd.DataFrame(
        get_prediction_results()
    )

    if predictions.empty or results.empty:

        st.info(
            "Kalibrasyon için sonuçlanmış "
            "tahmin gerekiyor."
        )

    else:

        df = predictions.merge(
            results[
                [
                    "prediction_id",
                    "result"
                ]
            ],
            left_on="id",
            right_on="prediction_id",
            how="inner"
        )

        df = df[
            df["result"].isin(
                [
                    "KAZANDI",
                    "KAYBETTİ"
                ]
            )
        ].copy()

        if len(df) < 10:

            st.info(
                f"Kalibrasyon için en az 10 "
                f"sonuç gerekli. Mevcut: {len(df)}"
            )

        else:

            df["actual"] = (
                df["result"]
                == "KAZANDI"
            ).astype(int)

            df["p"] = pd.to_numeric(
                df[
                    "predicted_probability"
                ],
                errors="coerce"
            )

            df = df.dropna(
                subset=["p"]
            )

            accuracy = (
                df["actual"].mean()
            )

            brier = (
                (
                    df["p"]
                    - df["actual"]
                ) ** 2
            ).mean()

            avg_prediction = (
                df["p"].mean()
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Accuracy",
                f"{accuracy * 100:.2f}%"
            )

            c2.metric(
                "Brier Score",
                f"{brier:.4f}"
            )

            c3.metric(
                "Ortalama Model Olasılığı",
                f"{avg_prediction * 100:.2f}%"
            )

            st.divider()

            st.subheader(
                "Kalibrasyon Verisini Güncelle"
            )

            for _, row in df.iterrows():

                prediction_id = row["id"]

                predicted_probability = safe_float(
                    row["p"]
                )

                actual_result = safe_float(
                    row["actual"]
                )

                prediction_error = (
                    actual_result
                    - predicted_probability
                )

                existing = fetch_table(
                    "calibration_data",
                    "*"
                )

                exists = any(
                    str(x.get("prediction_id"))
                    == str(prediction_id)
                    for x in existing
                )

                if not exists:

                    calibration = {
                        "prediction_id":
                            prediction_id,
                        "predicted_probability":
                            predicted_probability,
                        "actual_result":
                            actual_result,
                        "prediction_error":
                            prediction_error,
                        "model_version":
                            row.get(
                                "model_version",
                                MODEL_VERSION
                            )
                    }

                    insert_row(
                        "calibration_data",
                        calibration
                    )

            # Model settings
            existing_settings = fetch_table(
                "model_settings",
                "*"
            )

            parameters = {
                "sample_size":
                    sample_size,
                "min_ev":
                    min_ev,
                "min_confidence":
                    min_confidence,
                "method":
                    "Poisson",
                "model":
                    MODEL_VERSION
            }

            settings_data = {
                "model_version":
                    MODEL_VERSION,
                "parameters":
                    parameters,
                "calibration_sample_size":
                    len(df),
                "accuracy":
                    accuracy,
                "brier_score":
                    brier
            }

            if existing_settings:

                latest = existing_settings[-1]

                update_row(
                    "model_settings",
                    latest["id"],
                    settings_data
                )

            else:

                insert_row(
                    "model_settings",
                    settings_data
                )

            st.success(
                "✅ Kalibrasyon verileri güncellendi."
            )

            st.dataframe(
                df[
                    [
                        "market",
                        "selection",
                        "predicted_probability",
                        "actual"
                    ]
                ],
                use_container_width=True,
                hide_index=True
                            )
