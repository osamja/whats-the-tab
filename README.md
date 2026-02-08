# whats-the-tab

Developing: Start notebook from project directory
`jupyter notebook`

Then copy the server url, etc. http://localhost:8888/?token=ed084ae
In the ipython notebook python kernel, paste the url into the open by url option.

django app for transcribing music from audio files

## Local setup with uv

```bash
cd /home/samus/programming-projects/whats-the-tab
uv sync --frozen --no-dev --no-install-project
uv run python manage.py migrate
USE_PYTORCH=True IS_ASYNC=False uv run python manage.py runserver 0.0.0.0:8008
```

## Run with Docker

Build and run:

```bash
cd /home/samus/programming-projects/whats-the-tab
docker compose up --build
```

Stop:

```bash
docker compose down
```

The app is available at `http://127.0.0.1:8008` and Redis runs in a sidecar container.

go live checklist
- check @todo and remove anything that is not needed for production
- validate unicode characters for audio filenames in upload process https://docs.djangoproject.com/en/5.0/ref/validators/#validate-unicode-slug
- Give credit to mt3

Design Decisions
- Use Django for backend
- Because the transcription model takes up nearly 8GB (the size of our gtx 1080 vram), we will keep the transcribe api synchrounous.  We dont' have to worry about file name conflicts in the generation process, and can ensure the model does not get overloaded with requests.  Once we bring the rtx 4090 online, we can make the api async.
	make the apis async (once 4090 comes online)

		add a status field to model

		add a status api

		install dramatiq and redis message broker

# Bring transcribe server online
* Start django server

	Production
	`cd /home/sammy/workspace/whats-the-tab; source venv/bin/activate; gunicorn --bind 0.0.0.0:8008 musictranscription.wsgi`

	Development
	`cd /home/sammy/workspace/whats-the-tab; source venv/bin/activate; python manage.py runserver 0:8008`

* Start tailscale on desktop
`sudo tailscaled` or `sudo tailscale up`

* Start redis server
`redis-server`

* Start dramatiq task processor
`cd /home/sammy/workspace/whats-the-tab; source venv/bin/activate; python run_dramatiq.py transcribeapp.tasks`

# Setup new desktop
* Clone repo
* Copy soundfont file from google drive
* open `docs/index.ipynb` and run the commands from the Setup Environment cell in the beginning of the notebook

# Troubleshooting
* Shell into django server 
```
cd /home/sammy/workspace/whats-the-tab
source venv/bin/activate
python manage.py shell
```

# Django transcription backends
The transcription app runs on the PyTorch MT3 backend.

Backend selection is controlled by `USE_PYTORCH`:
- `USE_PYTORCH=True` uses `transcribeapp/ml_pytorch.py`

Recommended local development (sync mode for easier debugging):
```bash
cd /home/sammy/workspace/whats-the-tab
source venv/bin/activate
USE_PYTORCH=True IS_ASYNC=False python manage.py runserver 127.0.0.1:8008
```

# PyTorch checkpoint notes
PyTorch checkpoint is expected at:
`pytorch_mt3/mt3_pytorch_checkpoint.pt`

`PyTorchInferenceModel` resolves relative checkpoint paths against the project root, so
`pytorch_mt3/mt3_pytorch_checkpoint.pt` is valid when running Django from
the repository root.

# API smoke test
Run server first, then:

```bash
# 1) Upload audio
curl -F "audio=@/home/samus/programming-projects/whats-the-tab/dataset/in-the-morning-jcole.mp3" \
  http://127.0.0.1:8008/transcribe/upload/
```

From the response, copy `audio_midi_id` and use it below:

```bash
# 2) Generate MIDI
curl -X POST \
  -d "audio_midi_id=<AUDIO_MIDI_ID>" \
  -d "num_transcription_segments=1" \
  -d "audio_chunk_length=30" \
  http://127.0.0.1:8008/transcribe/generate/

# 3) Check status
curl http://127.0.0.1:8008/transcribe/status/<AUDIO_MIDI_ID>/

# 4) List available MIDI chunks
curl http://127.0.0.1:8008/transcribe/midi_chunks/<AUDIO_MIDI_ID>/

# 5) Download first MIDI chunk
curl -o /tmp/smoke_chunk0.midi \
  http://127.0.0.1:8008/transcribe/download_midi_chunk/<AUDIO_MIDI_ID>/0
```
