import os
import json
import joblib
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render, redirect
from .models import RiwayatPrediksi

load_dotenv()

MODEL_PATH = os.path.join(settings.BASE_DIR, "model_spotify.joblib")
CSV_PATH = os.path.join(settings.BASE_DIR, "Spotify-2000.csv")

model = joblib.load(MODEL_PATH)
df_spotify = pd.read_csv(CSV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def beranda(request):
    hasil_prediksi = None
    judul_terakhir = None
    active_tab = "dashboard"

    if request.method == "POST":
        active_tab = "prediksi"
        judul = request.POST.get("judul_lagu") or "Lagu Tanpa Judul"
        year = int(request.POST.get("year"))
        bpm = int(request.POST.get("bpm"))
        energy = int(request.POST.get("energy"))
        danceability = int(request.POST.get("danceability"))
        loudness = int(request.POST.get("loudness"))
        liveness = int(request.POST.get("liveness"))
        valence = int(request.POST.get("valence"))
        duration = float(request.POST.get("duration"))
        acousticness = int(request.POST.get("acousticness"))
        speechiness = int(request.POST.get("speechiness"))

        data_baru = pd.DataFrame({
            "Year": [year],
            "Beats Per Minute (BPM)": [bpm],
            "Energy": [energy],
            "Danceability": [danceability],
            "Loudness (dB)": [loudness],
            "Liveness": [liveness],
            "Valence": [valence],
            "Length (Duration)": [duration],
            "Acousticness": [acousticness],
            "Speechiness": [speechiness],
        })

        skor = round(float(model.predict(data_baru)[0]), 1)

        RiwayatPrediksi.objects.create(
            judul_lagu=judul,
            year=year,
            bpm=bpm,
            energy=energy,
            danceability=danceability,
            loudness=loudness,
            liveness=liveness,
            valence=valence,
            duration=duration,
            acousticness=acousticness,
            speechiness=speechiness,
            skor_popularitas=skor,
        )

        hasil_prediksi = skor
        judul_terakhir = judul

    riwayat = RiwayatPrediksi.objects.all().order_by("-tanggal_prediksi")

    # Statistik Ringkasan Dashboard
    total_dataset = len(df_spotify)
    rata_popularitas = round(float(df_spotify["Popularity"].mean()), 1)
    genre_teratas = str(df_spotify["Top Genre"].value_counts().index[0]).title()
    total_riwayat = riwayat.count()

    # Top 5 Lagu di Dataset Spotify-2000
    top_lagu_df = df_spotify.sort_values(by="Popularity", ascending=False).head(5)
    top_lagu = top_lagu_df[["Title", "Artist", "Top Genre", "Year", "Popularity"]].to_dict("records")

    # Grafik 1: Rata-rata Popularitas 10 Genre Terbanyak
    top_genres = df_spotify["Top Genre"].value_counts().head(10).index
    rata_genre = (
        df_spotify[df_spotify["Top Genre"].isin(top_genres)]
        .groupby("Top Genre")["Popularity"]
        .mean()
        .round(1)
        .sort_values(ascending=False)
    )

    # Grafik 2: Tren Popularitas per Tahun Rilis
    tren_tahun = (
        df_spotify.groupby("Year")["Popularity"]
        .mean()
        .round(1)
        .sort_index()
    )

    # Grafik 3: Faktor Penentu Popularitas
    nama_fitur = [
        "Tahun Rilis",
        "Kecepatan Ketukan (Tempo)",
        "Semangat Musik (Energi)",
        "Enak Buat Joget",
        "Kenyaringan Suara",
        "Nuansa Konser Live",
        "Suasana Ceria",
        "Durasi Lagu",
        "Nuansa Akustik",
        "Banyak Lirik / Rap"
    ]
    bobot_fitur = [round(float(x) * 100, 1) for x in model.feature_importances_]
    urutan_fitur = sorted(zip(nama_fitur, bobot_fitur), key=lambda x: x[1], reverse=True)
    fitur_labels = [str(x[0]) for x in urutan_fitur]
    fitur_values = [float(x[1]) for x in urutan_fitur]

    context = {
        "hasil_prediksi": hasil_prediksi,
        "judul_terakhir": judul_terakhir,
        "active_tab": active_tab,
        "riwayat": riwayat[:20],
        "total_dataset": total_dataset,
        "rata_popularitas": rata_popularitas,
        "genre_teratas": genre_teratas,
        "total_riwayat": total_riwayat,
        "top_lagu": top_lagu,
        "genre_labels": json.dumps([str(g) for g in rata_genre.index]),
        "genre_values": json.dumps([float(v) for v in rata_genre.values]),
        "tahun_labels": json.dumps([int(y) for y in tren_tahun.index]),
        "tahun_values": json.dumps([float(v) for v in tren_tahun.values]),
        "fitur_labels": json.dumps(fitur_labels),
        "fitur_values": json.dumps(fitur_values),
    }
    return render(request, "predictor/index.html", context)


def tanya_ai(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body)
            pertanyaan = body.get("pertanyaan", "").strip()

            if not pertanyaan:
                return JsonResponse({"jawaban": "Silakan ketik pertanyaan terlebih dahulu."})

            api_key = GEMINI_API_KEY.strip()
            if not api_key or "TEMPEL" in api_key:
                return JsonResponse({
                    "jawaban": "API Key Gemini belum diisi di file views.py."
                })

            instruksi_sistem = (
                "Kamu adalah asisten AI cerdas untuk aplikasi web 'Radar Lagu', sistem prediksi "
                "skor popularitas lagu Spotify (skala 0-100) berbasis Machine Learning Random Forest. "
                "Jawablah pertanyaan pengguna secara akurat, langsung ke inti pertanyaan, ramah, dan ringkas "
                "dalam bahasa Indonesia tanpa memakai format markdown bintang (**)."
            )

            client = genai.Client(api_key=api_key)
            daftar_model = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
            error_terakhir = ""

            for nama_model in daftar_model:
                try:
                    response = client.models.generate_content(
                        model=nama_model,
                        contents=pertanyaan,
                        config=types.GenerateContentConfig(
                            system_instruction=instruksi_sistem,
                            temperature=0.7,
                        ),
                    )
                    teks_bersih = response.text.replace("**", "")
                    return JsonResponse({"jawaban": teks_bersih})
                except Exception as err:
                    error_terakhir = str(err)
                    continue

            return JsonResponse({
                "jawaban": "Server AI Google sedang sangat sibuk (503). Silakan tunggu 5 detik lalu klik Kirim lagi ya!"
            })
        except Exception as e:
            return JsonResponse({"jawaban": f"Terjadi kendala sistem: {str(e)}"})

    return JsonResponse({"jawaban": "Metode tidak diizinkan."}, status=405)


def hapus_riwayat(request, id):
    item = RiwayatPrediksi.objects.get(id=id)
    item.delete()
    return redirect("beranda")
