# whats-the-tab

Developing
cd to the project directory and run:
```
jupyter notebook
```
Then copy the server url, etc. http://localhost:8888/?token=ed084ae
In the ipython notebook python kernel, paste the url into the open by url option.


django app for transcribing music from audio files


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
```
cd ~/whats-the-tab/src/mt3-transcription/musictranscription
source ../venv/bin/activate
python manage.py runserver 0:8008
```

* Start tailscale on desktop
`sudo tailscaled` or `sudo tailscale up`

* Start redis server
`redis-server`

* Start dramatiq
```
cd ~/whats-the-tab/src/mt3-transcription/musictranscription
source ../venv/bin/activate
python run_dramatiq.py transcribeapp.tasks
```

# Setup new desktop
* Clone repo
* Copy soundfont file from google drive
* open index.ipynb and run the commands from the Setup Environment cell in the beginning of the notebook

# Troubleshooting
* Shell into django server 
```
cd ~/whats-the-tab/src/mt3-transcription/musictranscription
source ../venv/bin/activate
python manage.py shell
```
