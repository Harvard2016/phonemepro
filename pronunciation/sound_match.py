"""Whether a practice take was an attempt at its target word, allowing for accent.

An exact match would drop strong accents: someone who says "tink" for "think" has
attempted every sound. So a target sound counts as attempted when the sound heard
in its place is the same broad class: any vowel for a vowel, or a consonant from
the same group (web/src/engine/sound_groups.json). A different word shares few
classes in the same places and scores low.

The web app runs the same rule in web/src/engine/soundMatch.js; both are pinned to
tests/fixtures/sound_match_cases.json. The threshold applied to this share was
measured by scripts/evaluate_match.py (Docs/model/match_threshold.json).
"""
import json
from decimal import ROUND_HALF_EVEN, Decimal
from functools import lru_cache
from pathlib import Path

from pronunciation.phonemes import UNSCORABLE_PHONEMES, strip_stress
from pronunciation.scoring import align

GROUPS_PATH = Path(__file__).resolve().parent.parent / "web" / "src" / "engine" / "sound_groups.json"


@lru_cache(maxsize=1)
def _classes() -> tuple[frozenset, dict]:
    data = json.loads(GROUPS_PATH.read_text(encoding="utf-8"))
    partners: dict[str, set] = {}
    for group in data["groups"]:
        for sound in group:
            partners.setdefault(sound, set()).update(group)
    return frozenset(data["vowels"]), partners


def same_class(target: str, heard: str) -> bool:
    """True when `heard` is an acceptable stand-in for `target` (stress ignored)."""
    target, heard = strip_stress(target), strip_stress(heard)
    if target == heard:
        return True
    vowels, partners = _classes()
    if target in vowels and heard in vowels:
        return True
    return heard in partners.get(target, ())


def sounds_attempted(target: list[str], heard: list[str]) -> int:
    """Percentage of target sounds attempted: heard as written, or as a sound of the same class."""
    if not target:
        return 0
    ref = [strip_stress(p) for p in target]
    hyp = [strip_stress(p) for p in heard]
    attempted = 0
    for op in align(ref, hyp).ops:
        if op.kind == "insert":
            continue
        if ref[op.ref_index] in UNSCORABLE_PHONEMES:
            attempted += 1  # the model cannot hear these, so they are given the benefit of the doubt
        elif op.kind != "delete" and same_class(ref[op.ref_index], hyp[op.hyp_index]):
            attempted += 1
    share = Decimal(attempted * 100) / Decimal(len(ref))
    return int(share.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
