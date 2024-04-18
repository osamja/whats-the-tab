from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_audio, name='upload_audio'),
    path('generate/', views.transcribe, name='transcribe'),
    path('download/<int:audio_id>/', views.download_midi, name='download_midi'),
    path('status/<int:audio_id>/', views.audio_status, name='audio_status'),
]
