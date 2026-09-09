import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ============================================================
# FUTBOL QUANT ENGINE - MOBİL İLK GOL ALARM SİSTEMİ
# Streamlit Cloud uyumlu sade sürüm
# ============================================================

st.set_page_config(
    page_title="Futbol Quant Engine",
    page_icon="⚽",
    layout="wide"
)

# ------------------------------------------------------------
# BAŞLIK
# ------------------------------------------------------------

st.title("⚽ FUTBOL QUANT ENGINE")
st.subheader("🔥 Mobil İlk Gol Alarm Sistemi")

st.info(
    "Bu sürüm harici makine öğrenmesi kütüphaneleri kullanmaz. "
    "CSV veya Excel verisini yükleyerek temel istatistiksel analiz yapar."
)

# ------------------------------------------------------------
# VERİ YÜKLEME
# ------------------------------------------------------------

st.header("📂 1. Veri Dosyası")

uploaded_file = st.file_uploader(
    "CSV veya Excel dosyanı yükle",
    type=["csv", "xlsx", "xls"]
)

df = None

if uploaded_file is not None:

    try:

        if uploaded_file.name.lower().endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(
                    uploaded_file,
                    sep=";",
                    encoding="utf-8"
                )

        else:
            df = pd.read_excel(uploaded_file)

        st.success(
            f"✅ Dosya başarıyla yüklendi: {uploaded_file.name}"
        )

        st.write(
            f"📊 Satır: **{len(df):,}** | "
            f"Sütun: **{len(df.columns)}**"
        )

        with st.expander("Verinin ilk 20 satırını göster"):
            st.dataframe(
                df.head(20),
                use_container_width=True
            )

    except Exception as e:

        st.error("❌ Dosya okunamadı.")

        st.code(str(e))

# ------------------------------------------------------------
# SÜTUNLARI GÖSTER
# ------------------------------------------------------------

if df is not None:

    st.header("🔎 2. Veri Yapısı")

    columns = list(df.columns)

    st.write("Bulunan sütunlar:")

    st.code(
        "\n".join(str(x) for x in columns)
    )

    # --------------------------------------------------------
    # SAYISAL SÜTUNLAR
    # --------------------------------------------------------

    numeric_columns = list(
        df.select_dtypes(include=np.number).columns
    )

    if numeric_columns:

        st.subheader("📈 Sayısal Sütunlar")

        st.write(
            ", ".join(str(x) for x in numeric_columns)
        )

        # ----------------------------------------------------
        # TEMEL İSTATİSTİK
        # ----------------------------------------------------

        st.header("📊 3. İstatistiksel Analiz")

        selected_columns = st.multiselect(
            "Analiz edilecek sütunları seç",
            numeric_columns,
            default=numeric_columns[:10]
        )

        if selected_columns:

            stats = df[selected_columns].describe().T

            stats["missing"] = (
                df[selected_columns]
                .isna()
                .sum()
            )

            stats["missing_%"] = (
                stats["missing"] /
                max(len(df), 1) *
                100
            )

            st.dataframe(
                stats.round(3),
                use_container_width=True
            )

    # --------------------------------------------------------
    # İLK GOL ANALİZİ
    # --------------------------------------------------------

    st.header("⚽ 4. İlk Gol Analizi")

    st.write(
        "Verinde dakika ve gol bilgisi varsa aşağıdaki alanları seç."
    )

    minute_candidates = [
        c for c in columns
        if any(
            x in str(c).lower()
            for x in [
                "minute",
                "min",
                "dakika",
                "dk"
            ]
        )
    ]

    goal_candidates = [
        c for c in columns
        if any(
            x in str(c).lower()
            for x in [
                "goal",
                "gol",
                "score",
                "skor"
            ]
        )
    ]

    if minute_candidates:

        minute_col = st.selectbox(
            "⏱️ Dakika sütunu",
            minute_candidates
        )

    else:

        minute_col = st.selectbox(
            "⏱️ Dakika sütunu",
            ["Seçiniz"] + columns
        )

    if goal_candidates:

        goal_col = st.selectbox(
            "⚽ Gol sütunu",
            goal_candidates
        )

    else:

        goal_col = st.selectbox(
            "⚽ Gol sütunu",
            ["Seçiniz"] + columns
        )

    if (
        minute_col != "Seçiniz"
        and goal_col != "Seçiniz"
    ):

        temp = df[[minute_col, goal_col]].copy()

        temp[minute_col] = pd.to_numeric(
            temp[minute_col],
            errors="coerce"
        )

        # Gol sütununu sayıya çevirmeyi dene
        temp[goal_col] = pd.to_numeric(
            temp[goal_col],
            errors="coerce"
        )

        temp = temp.dropna(
            subset=[minute_col]
        )

        if len(temp) > 0:

            st.subheader("⏱️ Dakika Dağılımı")

            # Dakikaları 5 dakikalık gruplara ayır
            temp["Dakika Grubu"] = (
                (temp[minute_col] // 5) * 5
            )

            distribution = (
                temp.groupby("Dakika Grubu")
                .size()
                .reset_index(name="Gözlem")
            )

            st.bar_chart(
                distribution.set_index(
                    "Dakika Grubu"
                )
            )

            st.dataframe(
                distribution,
                use_container_width=True
            )

# ------------------------------------------------------------
# MANUEL İLK GOL SİNYALİ
# ------------------------------------------------------------

st.header("🔥 5. Canlı İlk Gol Alarmı")

st.write(
    "Aşağıdaki değerleri canlı maçtan girerek "
    "basit baskı skoru oluşturabilirsin."
)

col1, col2, col3 = st.columns(3)

with col1:

    minute = st.number_input(
        "⏱️ Maç dakikası",
        min_value=0,
        max_value=130,
        value=60
    )

    shots = st.number_input(
        "🎯 Toplam şut",
        min_value=0,
        max_value=100,
        value=8
    )

with col2:

    shots_on_target = st.number_input(
        "🥅 İsabetli şut",
        min_value=0,
        max_value=50,
        value=3
    )

    corners = st.number_input(
        "🚩 Korner",
        min_value=0,
        max_value=50,
        value=3
    )

with col3:

    xg = st.number_input(
        "📊 xG",
        min_value=0.0,
        max_value=10.0,
        value=0.80,
        step=0.05
    )

    possession = st.number_input(
        "⚽ Topa sahip olma %",
        min_value=0.0,
        max_value=100.0,
        value=55.0,
        step=1.0
    )

# ------------------------------------------------------------
# BASKI SKORU
# ------------------------------------------------------------

pressure_score = 0.0

pressure_score += min(shots * 1.5, 20)
pressure_score += min(shots_on_target * 4, 20)
pressure_score += min(corners * 2, 10)
pressure_score += min(xg * 15, 25)

if possession >= 55:
    pressure_score += 5

if minute >= 60:
    pressure_score += 5

if minute >= 75:
    pressure_score += 5

pressure_score = min(
    pressure_score,
    100
)

st.subheader(
    f"🔥 Baskı Skoru: {pressure_score:.1f}/100"
)

if pressure_score >= 70:

    st.error(
        "🔔 🔥 YÜKSEK BASKI - İLK GOL ALARM 🔥"
    )

    st.write(
        "İstatistikler yüksek baskı bölgesine işaret ediyor."
    )

elif pressure_score >= 50:

    st.warning(
        "⚠️ ORTA/YÜKSELEN BASKI"
    )

else:

    st.success(
        "🟢 DÜŞÜK BASKI — ALARM YOK"
    )

# ------------------------------------------------------------
# VERİ İNDİRME
# ------------------------------------------------------------

if df is not None:

    st.header("💾 6. Veriyi İndir")

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "⬇️ CSV indir",
        data=csv_data,
        file_name="futbol_quant_veri.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------
# ALT BİLGİ
# ------------------------------------------------------------

st.divider()

st.caption(
    "Futbol Quant Engine | İstatistiksel analiz aracıdır. "
    "Tahminler garanti değildir."
)
