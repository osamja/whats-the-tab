from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_audio, name='upload_audio'),
    path('upload_from_yt/', views.upload_from_youtube, name='upload_audio'),
    path('generate/', views.transcribe, name='transcribe'),
    path('status/<int:audio_midi_id>/', views.audio_status, name='audio_status'),
    # add a new endpoint to get the list of midi chunks
    path('midi_chunks/<int:audio_midi_id>/', views.list_midi_chunks, name='midi_chunks'),
    # add a new endpoint to get a specified midi chunk 
    path('download_midi_chunk/<int:audio_midi_id>/<int:segment_index>', views.download_midi_chunk, name='download_midi_chunk'),
]
