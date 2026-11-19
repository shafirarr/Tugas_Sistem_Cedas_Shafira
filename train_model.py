import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Memuat dataset
df = pd.read_csv("Spotify-2000.csv")

# 2. Preprocessing: bersihkan tanda koma pada kolom durasi agar menjadi numerik
df["Length (Duration)"] = (
    df["Length (Duration)"].astype(str).str.replace(",", "").astype(float)
)

# 3. Menentukan Fitur (X) dan Target (y)
fitur = [
    "Year",
    "Beats Per Minute (BPM)",
    "Energy",
    "Danceability",
    "Loudness (dB)",
    "Liveness",
    "Valence",
    "Length (Duration)",
    "Acousticness",
    "Speechiness",
]
X = df[fitur]
y = df["Popularity"]

# 4. Membagi data latih & uji, lalu melatih model Random Forest
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluasi singkat
prediksi = model.predict(X_test)
mae = mean_absolute_error(y_test, prediksi)
print(f"Model berhasil dilatih! Rata-rata error (MAE): {mae:.2f} poin")

# 6. Membekukan model ke file .joblib
joblib.dump(model, "model_spotify.joblib")
print("Model tersimpan ke 'model_spotify.joblib'")