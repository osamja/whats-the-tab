"""Check that a Compose stack turns an audio upload into a readable MIDI file."""

import argparse
import sys
import time
from pathlib import Path

import note_seq
import requests


def check(base_url, audio, output, timeout):
    with requests.Session() as http:
        with http.get(f"{base_url}/ready/", timeout=10) as response:
            response.raise_for_status()

        with http.post(f"{base_url}/_result/", timeout=10) as response:
            if response.status_code != 403:
                raise RuntimeError("The result callback accepted a request without a worker token")

        with audio.open("rb") as source:
            response = http.post(
                f"{base_url}/upload/",
                files={"audio": (audio.name, source, "application/octet-stream")},
                timeout=30,
            )
        response.raise_for_status()
        job_id = response.json()["audio_midi_id"]

        start = time.monotonic()
        response = http.post(
            f"{base_url}/generate/", data={"audio_midi_id": job_id}, timeout=10
        )
        response.raise_for_status()

        previous = None
        while time.monotonic() - start < timeout:
            response = http.get(f"{base_url}/status/{job_id}/", timeout=10)
            response.raise_for_status()
            state = response.json()
            progress = (state["status"], state["current_chunk"], state["total_chunks"])
            if progress != previous:
                print(f"Job {job_id}: {progress[0]} ({progress[1]}/{progress[2]})", flush=True)
                previous = progress
            if state["status"] == "completed":
                break
            if state["status"] == "failed":
                raise RuntimeError(f"Job {job_id} failed: {state['error_message']}")
            time.sleep(0.5)
        else:
            raise RuntimeError(f"Job {job_id} did not complete within {timeout}s")

        if not state["has_midi"] or state["current_chunk"] != state["total_chunks"]:
            raise RuntimeError(f"Job {job_id} reported incomplete results")

        response = http.get(f"{base_url}/download_midi/{job_id}/", timeout=30)
        response.raise_for_status()
        sequence = note_seq.midi_to_note_sequence(response.content)
        if not sequence.notes:
            raise RuntimeError(f"Job {job_id} produced a MIDI file with no notes")
        output.write_bytes(response.content)
        print(f"Saved {len(sequence.notes)} notes to {output} in {time.monotonic() - start:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8008/transcribe")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    try:
        check(args.url.rstrip("/"), args.audio, args.output, args.timeout)
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(str(exc))
