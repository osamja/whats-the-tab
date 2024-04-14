# tasks.py
import datetime, uuid

def get_audio_filename():
   fileHash = uuid.uuid4()
   date = getDate()
   audio_filename = date + '-' + fileHash.hex + '.wav'
   return audio_filename

def getAudioDirectory():
  return 'content/audio/'

def getDate():
  date = str(datetime.datetime.now())
  date = date.replace(" ", "-")
  date = date.replace(":", "-")
  date = date.replace(".", "-")
  return date




