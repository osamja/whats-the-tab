"""
Python script to interact with the models
"""

import os
import numpy as np
import torch
from torch.autograd import Variable
from transcribeapp.models import AudioMIDI


# Write a function that takes in an ID for an AudioMIDI object and returns the fields
def get_audio_midi_fields(audio_midi_id):
    """
    Function to get the fields of an AudioMIDI object
    """
    audio_midi = AudioMIDI.objects.get(id=audio_midi_id)
    return audio_midi.audio_file, audio_midi.midi_file