from django.contrib import admin
from django.urls import path
from predictor import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.beranda, name='beranda'),
    path('tanya-ai/', views.tanya_ai, name='tanya_ai'),
    path('hapus/<int:id>/', views.hapus_riwayat, name='hapus_riwayat'),
]