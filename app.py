import streamlit as st
from supabase import create_client, Client
from datetime import datetime
import pandas as pd

# =========================================================
# FUTBOL QUANT ENGINE
# KAYIT + SONUÇ + KASA + GEÇMİŞ SİSTEMİ
# =========================================================

st.set_page_config(
    page_title="Futbol Quant Engine",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ FUTBOL QUANT ENGINE")
st.caption("Maç • Tahmin • Sonuç • Kasa • Geçmiş • Kalibrasyon")

# =========================================================
# SUPABASE BAĞLANTISI
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
    st.error(
        "Supabase bağlantısı yapılandırılmamış. "
        "Önce Streamlit Secrets bölümüne SUPABASE_URL "
        "ve SUPABASE_KEY eklenmelidir."
    )

# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================

def get_bets():
    """Kayıtlı bahisleri getir."""
    try:
        result = (
            supabase
            .table("bets")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )

        return result.data or []

    except Exception as e:
        st.error(f"Geçmiş kayıtlar alınamadı: {e}")
        return []


def calculate_profit(stake, odds, result):
    """Bahis sonucuna göre kâr/zarar hesapla."""

    stake = float(stake)
    odds = float(odds)

    if result == "KAZANDI":
        return round(stake * (odds - 1), 2)

    if result == "KAYBETTİ":
        return round(-stake, 2)

    if result == "İADE":
        return 0.0

    return 0.0


def save_bet(data):
    """Yeni bahis kaydı oluştur."""
    try:
        supabase.table("bets").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Kayıt sırasında hata oluştu: {e}")
        return False


def update_bet(bet_id, data):
    """Geçmiş kaydı düzelt."""
    try:
        (
            supabase
            .table("bets")
            .update(data)
            .eq("id", bet_id)
            .execute()
        )

        return True

    except Exception as e:
        st.error(f"Kayıt güncellenemedi: {e}")
        return False


def delete_bet(bet_id):
    """Geçmiş kaydı sil."""
    try:
        (
            supabase
            .table("bets")
            .delete()
            .eq("id", bet_id)
            .execute()
        )

        return True

    except Exception as e:
        st.error(f"Kayıt silinemedi: {e}")
        return False


# =========================================================
# ANA MENÜ
# =========================================================

menu = st.sidebar.radio(
    "MENÜ",
    [
        "🏠 Ana Sayfa",
        "➕ Yeni Maç",
        "🏁 Sonuç Gir",
        "📚 Geçmiş",
        "📊 Performans",
        "🧠 Kalibrasyon"
    ]
)

# =========================================================
# ANA SAYFA
# =========================================================

if menu == "🏠 Ana Sayfa":

    st.subheader("⚽ Futbol Quant Engine")

    bets = get_bets() if db_ok else []

    total = len(bets)

    won = len([
        x for x in bets
        if x.get("result") == "KAZANDI"
    ])

    lost = len([
        x for x in bets
        if x.get("result") == "KAYBETTİ"
    ])

    pending = len([
        x for x in bets
        if x.get("result") == "BEKLEMEDE"
    ])

    profit = sum(
        float(x.get("profit") or 0)
        for x in bets
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Toplam Maç", total)
    col2.metric("Kazandı", won)
    col3.metric("Kaybetti", lost)
    col4.metric("Kâr / Zarar", f"{profit:.2f} TL")

    st.divider()

    st.info(
        "Sistem; girilen maçları veritabanında saklar. "
        "Maç sonucu girildiğinde sonuç ve kâr/zarar hesabı yapılır."
    )


# =========================================================
# YENİ MAÇ
# =========================================================

elif menu == "➕ Yeni Maç":

    st.subheader("➕ Yeni Maç Kaydı")

    with st.form("new_bet"):

        col1, col2 = st.columns(2)

        with col1:
            home_team = st.text_input("Ev Sahibi")
            away_team = st.text_input("Deplasman")
            market = st.text_input(
                "Bahis / Tahmin",
                placeholder="Örn: MS 1"
            )

        with col2:
            odds = st.number_input(
                "Oran",
                min_value=1.01,
                value=1.85,
                step=0.01
            )

            stake = st.number_input(
                "Bahis Tutarı (TL)",
                min_value=0.0,
                value=100.0,
                step=10.0
            )

            model_probability = st.number_input(
                "Model Olasılığı (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=0.1
            )

        notes = st.text_area("Not")

        submitted = st.form_submit_button(
            "💾 MAÇI KAYDET",
            type="primary"
        )

    if submitted:

        if not home_team or not away_team or not market:

            st.warning(
                "Ev sahibi, deplasman ve tahmin alanlarını doldur."
            )

        elif not db_ok:

            st.error("Supabase bağlantısı yok.")

        else:

            data = {
                "home_team": home_team,
                "away_team": away_team,
                "market": market,
                "odds": float(odds),
                "stake": float(stake),
                "model_probability": float(model_probability),
                "result": "BEKLEMEDE",
                "profit": 0.0,
                "notes": notes
            }

            if save_bet(data):

                st.success(
                    f"✅ {home_team} - {away_team} kaydedildi."
                )

                st.rerun()


# =========================================================
# SONUÇ GİR
# =========================================================

elif menu == "🏁 Sonuç Gir":

    st.subheader("🏁 Maç Sonucu Gir")

    bets = get_bets() if db_ok else []

    pending = [
        x for x in bets
        if x.get("result") == "BEKLEMEDE"
    ]

    if not pending:

        st.info("Bekleyen maç bulunmuyor.")

    else:

        options = {
            f"{x.get('home_team')} - {x.get('away_team')} | "
            f"{x.get('market')} | "
            f"Oran {x.get('odds')}": x
            for x in pending
        }

        selected_text = st.selectbox(
            "Maç seç",
            list(options.keys())
        )

        selected = options[selected_text]

        st.write(
            f"**Bahis:** {selected.get('market')}"
        )

        st.write(
            f"**Bahis tutarı:** "
            f"{float(selected.get('stake') or 0):.2f} TL"
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

            profit = calculate_profit(
                selected.get("stake"),
                selected.get("odds"),
                result
            )

            update_data = {
                "result": result,
                "profit": profit,
                "settled_at": datetime.utcnow().isoformat()
            }

            if update_bet(
                selected.get("id"),
                update_data
            ):

                st.success(
                    f"Sonuç: {result} | "
                    f"Kâr/Zarar: {profit:.2f} TL"
                )

                st.rerun()


# =========================================================
# GEÇMİŞ
# =========================================================

elif menu == "📚 Geçmiş":

    st.subheader("📚 Maç Geçmişi")

    bets = get_bets() if db_ok else []

    if not bets:

        st.info("Henüz kayıt bulunmuyor.")

    else:

        df = pd.DataFrame(bets)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("✏️ Kayıt Düzenle / 🗑️ Sil")

        ids = [
            x.get("id")
            for x in bets
        ]

        selected_id = st.selectbox(
            "Kayıt seç",
            ids
        )

        selected = next(
            x for x in bets
            if x.get("id") == selected_id
        )

        new_result = st.selectbox(
            "Sonuç",
            [
                "BEKLEMEDE",
                "KAZANDI",
                "KAYBETTİ",
                "İADE"
            ],
            index=[
                "BEKLEMEDE",
                "KAZANDI",
                "KAYBETTİ",
                "İADE"
            ].index(
                selected.get("result", "BEKLEMEDE")
            )
        )

        new_stake = st.number_input(
            "Bahis Tutarı",
            value=float(selected.get("stake") or 0)
        )

        new_odds = st.number_input(
            "Oran",
            value=float(selected.get("odds") or 1.01),
            min_value=1.01
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button("💾 DÜZELT"):

                new_profit = calculate_profit(
                    new_stake,
                    new_odds,
                    new_result
                )

                data = {
                    "stake": new_stake,
                    "odds": new_odds,
                    "result": new_result,
                    "profit": new_profit
                }

                if update_bet(
                    selected_id,
                    data
                ):

                    st.success("Kayıt güncellendi.")
                    st.rerun()

        with col2:

            if st.button(
                "🗑️ SİL",
                type="secondary"
            ):

                if delete_bet(selected_id):

                    st.success("Kayıt silindi.")
                    st.rerun()


# =========================================================
# PERFORMANS
# =========================================================

elif menu == "📊 Performans":

    st.subheader("📊 Performans")

    bets = get_bets() if db_ok else []

    settled = [
        x for x in bets
        if x.get("result") in [
            "KAZANDI",
            "KAYBETTİ",
            "İADE"
        ]
    ]

    if not settled:

        st.info("Performans için sonuçlanmış maç gerekiyor.")

    else:

        won = len([
            x for x in settled
            if x.get("result") == "KAZANDI"
        ])

        lost = len([
            x for x in settled
            if x.get("result") == "KAYBETTİ"
        ])

        total_stake = sum(
            float(x.get("stake") or 0)
            for x in settled
        )

        total_profit = sum(
            float(x.get("profit") or 0)
            for x in settled
        )

        win_rate = (
            won / (won + lost) * 100
            if won + lost > 0
            else 0
        )

        roi = (
            total_profit / total_stake * 100
            if total_stake > 0
            else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Kazanan", won)
        c2.metric("Kaybeden", lost)
        c3.metric(
            "Kazanma %",
            f"{win_rate:.2f}%"
        )
        c4.metric(
            "ROI",
            f"{roi:.2f}%"
        )

        st.metric(
            "Toplam Kâr / Zarar",
            f"{total_profit:.2f} TL"
        )


# =========================================================
# KALİBRASYON
# =========================================================

elif menu == "🧠 Kalibrasyon":

    st.subheader("🧠 Model Kalibrasyonu")

    bets = get_bets() if db_ok else []

    settled = [
        x for x in bets
        if x.get("result") in [
            "KAZANDI",
            "KAYBETTİ"
        ]
        and x.get("model_probability") is not None
    ]

    if len(settled) < 10:

        st.info(
            "Kalibrasyon için en az 10 sonuçlanmış "
            "ve model olasılığı girilmiş kayıt gerekiyor."
        )

    else:

        rows = []

        for x in settled:

            probability = float(
                x.get("model_probability") or 0
            )

            actual = (
                1
                if x.get("result") == "KAZANDI"
                else 0
            )

            rows.append({
                "model_probability": probability,
                "actual": actual
            })

        calibration_df = pd.DataFrame(rows)

        predicted = (
            calibration_df["model_probability"].mean()
        )

        actual_rate = (
            calibration_df["actual"].mean() * 100
        )

        st.metric(
            "Ortalama Model Olasılığı",
            f"{predicted:.2f}%"
        )

        st.metric(
            "Gerçekleşen Kazanma Oranı",
            f"{actual_rate:.2f}%"
        )

        st.write(
            "Bu bölüm ilerleyen aşamada modelin "
            "kalibrasyon katsayılarını otomatik "
            "hesaplayacak şekilde geliştirilecektir."
                )
