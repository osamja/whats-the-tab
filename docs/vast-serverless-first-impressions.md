# First impressions of Vast Serverless

Pyaar.ai turns audio into MIDI. I used this migration to solve a problem with
the old deployment on a single rented GPU. When the rental went away, the
transcription service went with it. Now Linode holds the website, Redis queue,
database, and private files. Vast workers run MT3 inference and return the
MIDI. A dispatcher gives the worker a short-lived URL for the audio.

## What worked

Two RTX 3060 workers on separate hosts handled two simultaneous private
uploads, each returning a verified MIDI in roughly 3.4 seconds once warm.
A GTX 1080 Ti also completed the full flow in 3.89 seconds. The public
browser test verified upload, preview, and download. These checks show that
the same app can use different rentals and GPU models.

## What took work

Cold starts were unpredictable. One rental could not be restarted, and
another spent longer than the job deadline pulling the image and setting up
the runtime. The largest image layer alone was 6.15 GiB compressed. One GPU
reported idle while the dispatcher still could not connect to it. A replacement
passed once Linode could connect to the address Vast assigned and run inference.

The check I'd use again is a short upload from the real app through to a
downloadable MIDI. Worker status is useful for deciding when to run that check.
Recovery of the same in-flight job after a worker disappears still needs a
staging test.

## What it cost

On November 21, 2026 at 16:48 UTC, the Vast API showed five running rentals.
Two served the live queue. The other three were a verified GTX 1080 Ti
standby, another RTX 3060, and an RTX 4090 comparison rental.

| Running capacity | Listed hourly rate | 30-day projection |
| --- | ---: | ---: |
| Two serving RTX 3060s | $0.1490 | $107.28 |
| Three additional rentals | $0.5198 | $374.28 |
| All five | $0.6688 | $481.56 |

The projection assumes the listed rates and worker counts stay constant.
It excludes Linode and object storage. Vast billing records for the project
from November 4 through November 21 at 00:00 UTC showed $8.30 in posted
charges, including $5.78 for GPU time and $2.53 for rental disk. That
billing window closes before the live rate snapshot.

The biggest cost lesson was to check the whole account after testing. The
RTX 4090 comparison rental alone was listed at $0.3711 per hour. A warm
capacity target also keeps a worker running when the queue is empty, so a
quiet app can still have a steady bill. [Vast pricing](https://docs.vast.ai/guides/pricing) · [Vast scaling](https://docs.vast.ai/guides/serverless/managing-scale)

## Feedback for the Serverless team

1. **Make worker reachability visible.** Vast assigned an idle worker, but
   Linode couldn't connect to its inference address. Expose a reachability
   check for the advertised address and port, and include the underlying
   socket error in the SDK's diagnostic output.
2. **Show cold-start progress.** One worker exceeded our 10-minute job
   deadline during setup. Show time spent pulling the image, installing the
   runtime, and loading the model so operators can decide when to wait or
   replace a worker.
3. **Show capacity and cost together.** An unavailable rental lingered
   while we waited for a replacement. Later, five rentals were running
   while two served the live queue. Show which workers count against
   capacity, what they cost per hour, and whether a requested stop has
   taken effect.

Reachability is the first thing I'd improve. A worker that appears ready
but can't accept a request was the hardest failure to diagnose.

[Architecture diagram](transcription-flow.svg) · [Test and cutover evidence](handoff-linode-cutover.md)
