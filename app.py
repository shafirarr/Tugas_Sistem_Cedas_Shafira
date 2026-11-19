import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Popularitas Lagu Spotify", page_icon="🎵", layout="wide")

# 1. Judul & Identitas Kelompok
st.title("🎵 Prediksi Skor Popularitas Lagu Spotify")
st.write("**Mini Proyek Sistem Cerdas**")
st.write("Oleh: Shafira Rahmaningtyas")
st.divider()

# 2. Memuat Model & Data dengan Cache
@st.cache_resource
def muat_model():
    return joblib.load("model_spotify.joblib")

@st.cache_data
def muat_data():
    df = pd.read_csv("Spotify-2000.csv")
    df["Length (Duration)"] = df["Length (Duration)"].astype(str).str.replace(",", "").astype(float)
    return df

model = muat_model()
df = muat_data()

# 3. Form Input Fitur Audio
st.subheader("🎛️ Masukkan Karakteristik Audio Lagu")

judul_lagu = st.text_input("Judul Lagu (Opsional - hanya untuk tampilan)", "Lagu Baru Saya")

col1, col2 = st.columns(2)

with col1:
    year = st.slider("Tahun Rilis (Year)", 1956, 2026, 2015)
    bpm = st.slider("Tempo / Beats Per Minute (BPM)", 37, 210, 120)
    energy = st.slider("Energy (1 - 100)", 1, 100, 65)
    danceability = st.slider("Danceability (1 - 100)", 1, 100, 60)
    loudness = st.slider("Loudness (dB)", -27, 0, -8)

with col2:
    liveness = st.slider("Liveness (1 - 100)", 1, 100, 15)
    valence = st.slider("Valence / Keceriaan (1 - 100)", 1, 100, 50)
    duration = st.number_input("Durasi Lagu (Detik)", min_value=60, max_value=1500, value=210)
    acousticness = st.slider("Acousticness (1 - 100)", 0, 100, 25)
    speechiness = st.slider("Speechiness (1 - 100)", 1, 60, 5)

# 4. Tombol & Proses Prediksi
if st.button("🚀 Prediksi Popularitas", type="primary"):
    data_baru = pd.DataFrame({
        "Year": [year],
        "Beats Per Minute (BPM)": [bpm],
        "Energy": [energy],
        "Danceability": [danceability],
        "Loudness (dB)": [loudness],
        "Liveness": [liveness],
        "Valence": [valence],
        "Length (Duration)": [float(duration)],
        "Acousticness": [acousticness],
        "Speechiness": [speechiness],
    })

    hasil = model.predict(data_baru)[0]

    st.success(f"Perkiraan Skor Popularitas untuk **'{judul_lagu}'**: **{hasil:.1f} / 100**")
    if hasil >= 70:
        st.info("🔥 Lagu ini memiliki potensi tinggi untuk menjadi hits populer!")
    elif hasil >= 50:
        st.info("👍 Lagu ini memiliki tingkat popularitas menengah.")
    else:
        st.warning("🎧 Lagu ini cenderung segmented / niche.")

st.divider()

# 5. Grafik Eksplorasi Data
st.subheader("📊 Rata-Rata Skor Popularitas pada 10 Genre Terbanyak")
top_10_genres = df["Top Genre"].value_counts().head(10).index
df_top = df[df["Top Genre"].isin(top_10_genres)]
rata_genre = df_top.groupby("Top Genre")["Popularity"].mean().sort_values(ascending=False)

st.bar_chart(rata_genre)