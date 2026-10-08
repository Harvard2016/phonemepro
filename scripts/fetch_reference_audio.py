"""Download human reference recordings for the practice words from Wikimedia Commons.

    python -m scripts.fetch_reference_audio

Commons hosts volunteer recordings named En-us-<word>.ogg, En-uk-<word>.ogg and
En-au-<word>.ogg. Each one found is converted to MP3 (so Safari can play it) and
saved to web/public/audio/<accent>/<word>.mp3, with the speaker, licence and
source page recorded in web/public/audio/credits.json. Words with no recording
in an accent fall back to the browser's synthetic voice in the app.

Requires ffmpeg.
"""
import json
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
WORDS_PATH = ROOT_DIR / "web" / "public" / "lexicon" / "practice_words.json"
AUDIO_DIR = ROOT_DIR / "web" / "public" / "audio"
CREDITS_PATH = AUDIO_DIR / "credits.json"
API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "PhonemePro/1.0 (https://github.com/Harvard2016/phonemepro; pronunciation practice app)"
PREFIXES = {"ga": ["En-us-"], "rp": ["En-uk-", "En-gb-"], "au": ["En-au-"]}
# Licences that allow redistribution with attribution.
ALLOWED_LICENSE = re.compile(r"^(cc0|cc[ -]by|public domain|pd)", re.IGNORECASE)


def api_get(params: dict) -> dict:
    url = f"{API}?{urllib.parse.urlencode({**params, 'format': 'json'})}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", value or "")).strip()


def find_recording(word: str, accent: str) -> dict | None:
    titles = [f"File:{prefix}{word}.{ext}" for prefix in PREFIXES[accent] for ext in ("ogg", "oga", "wav")]
    data = api_get({
        "action": "query", "titles": "|".join(titles), "prop": "imageinfo",
        "iiprop": "url|extmetadata",
    })
    for page in data["query"]["pages"].values():
        if "imageinfo" not in page:
            continue
        info = page["imageinfo"][0]
        meta = info.get("extmetadata", {})
        license_name = meta.get("LicenseShortName", {}).get("value", "")
        if not ALLOWED_LICENSE.match(license_name):
            continue
        return {
            "url": info["url"],
            "page": info["descriptionurl"],
            "title": page["title"].removeprefix("File:"),
            "author": strip_html(meta.get("Artist", {}).get("value", "")) or "Wikimedia Commons contributor",
            "license": license_name,
            "license_url": meta.get("LicenseUrl", {}).get("value", ""),
        }
    return None


def download_as_mp3(url: str, destination: Path):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with tempfile.NamedTemporaryFile(suffix=Path(url).suffix) as source:
        with urllib.request.urlopen(request, timeout=60) as response:
            source.write(response.read())
        source.flush()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Mono, trimmed of leading silence, loudness-normalized so clips match each other.
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", source.name, "-ac", "1", "-ar", "44100",
             "-af", "silenceremove=start_periods=1:start_threshold=-45dB,loudnorm=I=-18:TP=-2",
             "-codec:a", "libmp3lame", "-b:a", "64k", str(destination)],
            check=True,
        )


def main():
    levels = json.loads(WORDS_PATH.read_text())
    words = [word for level in levels.values() for word in level]
    credits = json.loads(CREDITS_PATH.read_text()) if CREDITS_PATH.exists() else {}

    for accent in PREFIXES:
        credits.setdefault(accent, {})
        for word in words:
            destination = AUDIO_DIR / accent / f"{word}.mp3"
            if word in credits[accent] and destination.exists():
                continue
            recording = find_recording(word, accent)
            time.sleep(0.3)
            if recording is None:
                continue
            download_as_mp3(recording.pop("url"), destination)
            recording["changes"] = "converted to MP3, trimmed, loudness normalized"
            credits[accent][word] = recording
            time.sleep(0.3)
        print(f"{accent}: {len(credits[accent])} of {len(words)} words have a human recording", flush=True)

    CREDITS_PATH.write_text(json.dumps(credits, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {CREDITS_PATH}")


if __name__ == "__main__":
    main()
