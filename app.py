import streamlit as st

st.set_page_config(
    page_title="Futbol Quant Engine",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ FUTBOL QUANT ENGINE")
st.caption("Quant Analysis System V5.0")

st.divider()

st.subheader("📁 MAÇ VERİLERİ")

home_file = st.file_uploader(
    "Ev Sahibi Takım Dosyası",
    type=["xlsx", "xls", "csv"]
)

away_file = st.file_uploader(
    "Deplasman Takım Dosyası",
    type=["xlsx", "xls", "csv"]
)

league_file = st.file_uploader(
    "Lig / Rakip Kalitesi Dosyası",
    type=["xlsx", "xls", "csv"]
)

odds_file = st.file_uploader(
    "Bahis Oranları Dosyası",
    type=["xlsx", "xls", "csv"]
)

st.divider()

if st.button("🔬 ANALİZİ BAŞLAT", type="primary"):

    if not home_file or not away_file or not league_file or not odds_file:
        st.error("Lütfen 4 veri dosyasının tamamını yükleyin.")

    else:
        st.success("Dosyalar alındı.")

        st.info(
            "Analiz motoru henüz aktif değil. "
            "Bir sonraki aşamada veri motoru eklenecek."
        )
