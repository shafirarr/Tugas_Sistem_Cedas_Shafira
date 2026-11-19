from django.db import models

class RiwayatPrediksi(models.Model):
    judul_lagu = models.CharField(max_length=150)
    year = models.IntegerField()
    bpm = models.IntegerField()
    energy = models.IntegerField()
    danceability = models.IntegerField()
    loudness = models.IntegerField()
    liveness = models.IntegerField()
    valence = models.IntegerField()
    duration = models.FloatField()
    acousticness = models.IntegerField()
    speechiness = models.IntegerField()
    skor_popularitas = models.FloatField()
    tanggal_prediksi = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.judul_lagu} - Skor: {self.skor_popularitas}"