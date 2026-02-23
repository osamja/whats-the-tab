from django.urls import path
from . import views

urlpatterns = [
    path('', views.health, name='health'),
    path('upload/', views.upload_audio, name='upload_audio'),
    path('upload_from_yt/', views.upload_from_youtube, name='upload_audio'),
    path('generate/', views.transcribe, name='transcribe'),
    path('status/<int:audio_midi_id>/', views.audio_status, name='audio_status'),
    path('midi/<int:audio_midi_id>/', views.get_midi, name='get_midi'),
    path('download_midi/<int:audio_midi_id>/', views.download_midi, name='download_midi'),
]
