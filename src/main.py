import librosa

# 1. Get the file path to an included audio example

filename = 'music/Better-Together-Short_3.mp3'

# waveform y, and sampling rate sr
y, sr = librosa.load(filename)

# 2. Run the default beat tracker 
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# 3. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# plot the beat events
import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(15, 5))
librosa.display.waveshow(y, sr=sr, alpha=0.4)
plt.vlines(beat_times, -1, 1, color='r')
plt.show()
