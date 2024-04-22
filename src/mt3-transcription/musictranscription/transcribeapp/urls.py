from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_audio, name='upload_audio'),
    path('upload_from_yt/', views.upload_from_youtube, name='upload_audio'),
    path('generate/', views.transcribe, name='transcribe'),
    path('download/<int:audio_midi_id>/', views.download_midi, name='download_midi'),
    # allow download of midi wav file
    path('download_wav/<int:audio_midi_id>/', views.download_midi_wav, name='download_midi_wav'),
    path('status/<int:audio_midi_id>/', views.audio_status, name='audio_status'),
]
