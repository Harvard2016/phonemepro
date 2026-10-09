"""Fetch the synthetic voices the web app uses when a word has no human recording.

    python -m scripts.fetch_voices [--cache DIR]

The browser's own voices differ on every device (and are missing for most accents on
most devices), so the app carries one Piper voice per accent and runs it with the same
ONNX runtime as the phoneme model. Each voice is split into chunks like the phoneme model
and described in web/public/voices/voices.json.

Voices were chosen by ear and by feeding their output to the phoneme model. Note that the
`espeak` voice below is the phoneme set the text is converted with: several American Piper
voices were trained on British phoneme strings and say "tomato" the British way, so the
American voice here is one trained on American ones.
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "web" / "public" / "voices"
CHUNK_BYTES = 40 * 1024 * 1024
PIPER = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en"

VOICES = {
    "ga": {
        "file": "en_US-joe-medium",
        "url": f"{PIPER}/en_US/joe/medium/en_US-joe-medium.onnx",
        "espeak": "en-us",
        "speaker": None,
        "name": "Joe",
        "licence": "CC0",
        "source": "https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/joe/medium",
    },
    "rp": {
        "file": "en_GB-cori-medium",
        "url": f"{PIPER}/en_GB/cori/medium/en_GB-cori-medium.onnx",
        "espeak": "en-gb-x-rp",
        "speaker": None,
        "name": "Cori",
        "licence": "Public domain",
        "source": "https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_GB/cori/medium",
    },
    "au": {
        "file": "en_GB-vctk-medium",
        "url": f"{PIPER}/en_GB/vctk/medium/en_GB-vctk-medium.onnx",
        "espeak": "en-gb-x-rp",
        "speaker": "p326",  # one of the two Australian speakers in VCTK
        "name": "VCTK speaker p326",
        "licence": "CC BY 4.0",
        "source": "https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_GB/vctk/medium",
    },
}


def fetch(url: str, path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT_DIR / "data" / "voices")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*.part*"):
        stale.unlink()

    manifest = {}
    for accent, voice in VOICES.items():
        model = fetch(voice["url"], args.cache / f"{voice['file']}.onnx").read_bytes()
        config = json.loads(fetch(voice["url"] + ".json", args.cache / f"{voice['file']}.onnx.json").read_text())
        chunks = []
        for index in range(0, len(model), CHUNK_BYTES):
            name = f"{voice['file']}.onnx.part{index // CHUNK_BYTES:02d}"
            (OUT_DIR / name).write_bytes(model[index:index + CHUNK_BYTES])
            chunks.append(name)
        inference = config["inference"]
        manifest[accent] = {
            "name": voice["name"],
            "licence": voice["licence"],
            "source": voice["source"],
            "version": hashlib.sha256(model).hexdigest()[:12],
            "bytes": len(model),
            "chunks": chunks,
            "espeak": voice["espeak"],
            "speakerId": config["speaker_id_map"][voice["speaker"]] if voice["speaker"] else None,
            "sampleRate": config["audio"]["sample_rate"],
            "scales": [inference["noise_scale"], inference["length_scale"], inference["noise_w"]],
            "phonemeIds": config["phoneme_id_map"],
        }
        print(f"{accent}: {voice['file']} {len(model) / 1e6:.1f} MB in {len(chunks)} chunks")

    (OUT_DIR / "voices.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(f"Wrote {OUT_DIR / 'voices.json'}")


if __name__ == "__main__":
    main()
