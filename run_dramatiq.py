import os
import django
from dramatiq import cli
import sys

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "musictranscription.settings")
django.setup()

if __name__ == "__main__":
    sys.argv.extend(["--processes", "1", "--threads", "1"])
    cli.main()
