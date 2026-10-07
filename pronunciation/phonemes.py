"""ARPABET helpers and the articulation tips shown to learners."""

STRESS_DIGITS = "012"

# The fine-tuned model never emits these (0 of 45,798 predictions on the SpeechOcean762
# test split; each is rare in the training data), so it cannot judge them either way.
UNSCORABLE_PHONEMES = frozenset({"ZH", "OY"})
UNSCORABLE_TIP = "The model cannot judge this sound yet, so it is not counted."

PHONEME_TIPS = {
    "AA": "Open your mouth wide, like saying 'ah' at the doctor.",
    "AE": "Open your mouth and spread your lips slightly, like 'cat'.",
    "AH": "Relax your mouth - a short, neutral 'uh' sound.",
    "AO": "Round your lips slightly and say 'aw' as in 'thought'.",
    "AW": "Start with 'ah' and glide to 'oo', like 'cow'.",
    "AY": "Start with 'ah' and glide to 'ee', like 'my'.",
    "B": "Press your lips together, then release with a burst of air.",
    "CH": "Touch the tip of your tongue to the roof of your mouth, then release - 'ch' as in 'cheese'.",
    "D": "Touch your tongue to the ridge behind your top teeth, then release.",
    "DH": "Put your tongue between your teeth and vibrate - 'th' as in 'the'.",
    "EH": "Open your mouth slightly and say 'eh' as in 'bed'.",
    "ER": "Curl your tongue slightly back - 'er' as in 'bird'.",
    "EY": "Start with 'eh' and glide to 'ee' - 'ay' as in 'say'.",
    "F": "Lightly bite your lower lip and push air through.",
    "G": "Touch the back of your tongue to the soft palate, then release.",
    "HH": "Simply breathe out through your open mouth.",
    "IH": "Relax your tongue slightly - 'ih' as in 'sit'.",
    "IY": "Spread your lips and raise the front of your tongue - 'ee' as in 'see'.",
    "JH": "Touch your tongue to the roof of your mouth and release - 'j' as in 'judge'.",
    "K": "Touch the back of your tongue to the soft palate, then release with a puff.",
    "L": "Touch the tip of your tongue to the ridge behind your top teeth.",
    "M": "Press your lips together and hum.",
    "N": "Touch your tongue to the ridge behind your top teeth and hum.",
    "NG": "Touch the back of your tongue to the soft palate and hum - 'ng' as in 'sing'.",
    "OW": "Round your lips and glide from 'oh' to 'oo' - 'oh' as in 'go'.",
    "OY": "Round your lips for 'aw' then glide to 'ee' - 'oy' as in 'boy'.",
    "P": "Press your lips together, then release with a puff of air.",
    "R": "Curl your tongue slightly back without touching the roof.",
    "S": "Place your tongue close to the ridge behind your top teeth and push air through.",
    "SH": "Round your lips slightly and push air through - 'sh' as in 'she'.",
    "T": "Touch your tongue to the ridge behind your top teeth, then release sharply.",
    "TH": "Put your tongue between your teeth and push air - 'th' as in 'think'.",
    "UH": "Round your lips slightly - 'uh' as in 'book'.",
    "UW": "Round your lips tightly - 'oo' as in 'food'.",
    "V": "Lightly bite your lower lip and vibrate.",
    "W": "Round your lips tightly and then open - 'w' as in 'we'.",
    "Y": "Raise the front of your tongue toward the roof - 'y' as in 'yes'.",
    "Z": "Like 'S', but add vibration from your vocal cords.",
    "ZH": "Round your lips and push air with vibration - 'zh' as in 'measure'.",
}

CLARITY_COACHING = {
    "R": "Make the American R tighter and more centered. Avoid a tapped or rolled sound.",
    "ER": "Hold the vowel-r color together as one smooth sound instead of splitting it apart.",
    "IH": "Keep /IH/ short and relaxed, not stretched toward /IY/.",
    "IY": "Keep /IY/ high and tense, like a clear 'ee' sound.",
    "AE": "Open the jaw more for /AE/ and keep it bright, not flat.",
    "AH": "Use a short, central /AH/ instead of a broad open vowel.",
    "AA": "Keep /AA/ open and steady without drifting toward /AH/ or /AO/.",
    "AO": "Round /AO/ slightly so it does not flatten out.",
    "UH": "Keep /UH/ short and rounded, not tense like /UW/.",
    "UW": "Push the lips forward more for /UW/ and keep it tense.",
    "TH": "Keep the tongue lightly between the teeth for /TH/, not behind them.",
    "DH": "For /DH/, keep the tongue between the teeth and add voicing.",
    "V": "Let the lower lip touch the upper teeth and keep the sound voiced.",
    "W": "Round the lips strongly for /W/ and release quickly.",
    "T": "Make /T/ clean and crisp without adding an extra vowel after it.",
    "D": "Keep /D/ short and voiced without turning it into a flap or extra syllable.",
}

def strip_stress(phoneme: str) -> str:
    """Drop the lexical stress digit: ``AE1`` -> ``AE``."""
    return phoneme.rstrip(STRESS_DIGITS)


def stress_of(phoneme: str) -> str | None:
    """Return the stress digit of a vowel, or None for consonants."""
    return phoneme[-1] if phoneme and phoneme[-1] in STRESS_DIGITS else None


def get_tip(phoneme: str) -> str:
    """Get pronunciation tip for a phoneme (stress digit ignored)."""
    return PHONEME_TIPS.get(strip_stress(phoneme), f"Focus on clearly articulating '{phoneme}'.")


def build_clarity_coaching(target_phonemes: list[str]) -> list[str]:
    """Return targeted coaching tips for words that were recognized but not spoken clearly enough."""
    coaching = []
    seen = set()

    for phoneme in target_phonemes:
        tip = CLARITY_COACHING.get(strip_stress(phoneme))
        if tip and tip not in seen:
            coaching.append(tip)
            seen.add(tip)

    if not coaching:
        coaching.append("Slow the word down and keep each sound clean without adding extra vowel sounds.")

    return coaching[:3]
