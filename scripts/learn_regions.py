"""Learn regional accents from contributed takes without storing them.

    python -m scripts.learn_regions fit-projection        # once: the 256-number summary the browser computes
    python -m scripts.learn_regions simulate              # check the idea on VCTK
    python -m scripts.learn_regions ingest <file|folder>  # fold exported takes into the running totals
    python -m scripts.learn_regions ingest <file|folder> --dry-run   # check and summarise, write nothing
    python -m scripts.learn_regions ingest <folder> --rebuild        # pilot: totals from exactly these files
    python -m scripts.learn_regions reference             # native reference accents from VCTK
    python -m scripts.learn_regions report                # what the totals can tell so far

When someone opts in, the web app reduces each clear take to 256 numbers: the
mean and spread of an encoder layer over the take, standardized and projected
onto fixed directions (fit-projection). It is not audio and cannot be played
back, though like any voice-derived summary it may be characteristic of the
speaker, so it is still treated as personal data until it is folded in.

`ingest` reads files exported from the app's "Your data" page, checks every take,
and adds each vector to running totals (pronunciation/region_totals.py). Only a
take in the speaker's normal voice joins the totals for their country, and for
their region when one was given. Every practice take is an imitation of its
target accent, so it joins separate "country attempting accent" totals.
Only the totals and a list of take ids already counted are kept. `simulate` runs
the same path on VCTK speakers to show that a classifier built from totals alone works.

During a pilot, where people send their export files by hand, keep the files in a
private folder and run `ingest <folder> --rebuild` each time. The totals are then
built from exactly the files present, so removing someone's file and rebuilding
removes them from the totals, and contributors are counted correctly. `report`
then shows which regions have enough normal-voice takes to be told apart, and
how close each country's attempts at an accent land to native speakers of it
(`reference` builds those native reference accents from VCTK).
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from pronunciation.region_totals import (
    FEATURES,
    Totals,
    build_classifier,
    load_totals,
    mean_squared_distance,
    save_totals,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
PROJECTION_PATH = ROOT_DIR / "results" / "accent_projection.npz"
TOTALS_PATH = ROOT_DIR / "results" / "region_totals.npz"
ATTEMPTS_PATH = ROOT_DIR / "results" / "attempt_totals.npz"
REFERENCE_PATH = ROOT_DIR / "results" / "reference_totals.npz"
SUMMARY_PATH = ROOT_DIR / "results" / "region_summary.json"
# VCTK accent groups that stand in for native speakers of a practice accent until real
# normal-voice contributions exist. VCTK has two Australian speakers, too few to use.
REFERENCE_ACCENTS = {"ga": "american", "rp": "english_southern"}
SEEN_PATH = ROOT_DIR / "results" / "region_totals_seen.txt"
REPORT_PATH = ROOT_DIR / "Docs" / "model" / "eval_results.json"
MODEL_META_PATH = ROOT_DIR / "web" / "public" / "model" / "model.json"

UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
COUNTRY = re.compile(r"^[A-Za-z]{2}$")
EXPORT_VERSION = 2
ACCENTS = ("ga", "rp", "au")
# Broad regions a contributor may pick, only for countries where accent varies a lot
# by region. A take carries one of these codes or nothing; typed text is never accepted.
# The web app holds the same table (web/src/lib/regions.js).
REGIONS = {
    "US": {"NE": "Northeast", "S": "South", "MW": "Midwest", "W": "West"},
    "GB": {
        "ENG_S": "England (South)", "ENG_N": "England (North)", "SCT": "Scotland",
        "WLS": "Wales", "NIR": "Northern Ireland",
    },
    "AU": {"E": "Eastern states and Tasmania", "SA": "South Australia", "W": "Western Australia and the Northern Territory"},
    "CA": {"ATL": "Atlantic provinces", "QC": "Quebec", "ON": "Ontario", "W": "Prairies, British Columbia and the North"},
    "IE": {"E": "East (Leinster)", "S": "South (Munster)", "W": "West and north (Connacht, Ulster)"},
    "IN": {"N": "North", "S": "South", "E": "East and Northeast", "W": "West and Central"},
}
# The practice accent that is the standard of a country. It only decides the stored
# label `matches_home_accent`, which is checked here but plays no part in routing. The web app
# holds the same table (web/src/engine/contributions.js); both are pinned to
# tests/fixtures/contribution_export.json.
HOME_ACCENT = {"US": "ga", "GB": "rp", "AU": "au"}
# One person should not outweigh a region, so only this many takes count per contributor per run.
MAX_TAKES_PER_CONTRIBUTOR = 40
# Projected features are standardized, so honest values are small. Larger ones are malformed or hostile.
MAX_ABS_FEATURE = 60.0
MIN_CONTRIBUTORS = 8
MIN_TAKES = 200


def fit_projection():
    """Standardization and principal directions of the accent layer statistics, from VCTK training speakers."""
    import torch

    from scripts.accent_data import ACCENT_SPEAKERS, PROBE_LAYERS, PROBE_SPEAKERS, test_speakers
    from scripts.train_accent_head import HEAD_PATH, load_stats
    layer = torch.load(HEAD_PATH)["layer"]
    layer_index = PROBE_LAYERS.index(layer)

    windows = []
    for speakers in ACCENT_SPEAKERS.values():
        for speaker in sorted(set(speakers) - test_speakers(speakers)):
            stats = load_stats(speaker, layer_index)
            windows.append(stats[:, 1:].reshape(-1, stats.shape[-1]))
    data = np.concatenate(windows).astype(np.float64)

    mean = data.mean(axis=0)
    scale = data.std(axis=0) + 1e-6
    _, _, directions = np.linalg.svd((data - mean) / scale, full_matrices=False)
    PROJECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PROJECTION_PATH, mean=mean, scale=scale, directions=directions[:FEATURES].T, layer=layer)
    del PROBE_SPEAKERS  # imported for symmetry with the other scripts; probes are never used to fit
    print(f"Wrote {PROJECTION_PATH}: layer {layer}, {data.shape[1]} statistics -> {FEATURES} features, "
          f"from {len(data)} one-second windows")


def project(stats: np.ndarray) -> np.ndarray:
    projection = np.load(PROJECTION_PATH)
    return ((stats - projection["mean"]) / projection["scale"]) @ projection["directions"]


def simulate():
    """Treat VCTK speakers as contributors and compare a totals-only classifier with the trained accent head."""
    from scripts.accent_data import ACCENT_SPEAKERS, PROBE_LAYERS, test_speakers
    from scripts.train_accent_head import CLASS_NAMES, CLASSES, load_stats

    layer_index = PROBE_LAYERS.index(int(np.load(PROJECTION_PATH)["layer"]))
    totals = {accent: Totals.empty() for accent in CLASSES}
    tests = {"one_second": ([], []), "full_utterance": ([], [])}

    for accent in CLASSES:
        held_out = test_speakers(ACCENT_SPEAKERS[accent])
        for speaker in ACCENT_SPEAKERS[accent]:
            stats = load_stats(speaker, layer_index)
            windows = project(stats[:, 1:].reshape(-1, stats.shape[-1]))
            if speaker in held_out:
                tests["one_second"][0].append(windows)
                tests["one_second"][1].extend([accent] * len(windows))
                tests["full_utterance"][0].append(project(stats[:, 0]))
                tests["full_utterance"][1].extend([accent] * len(stats))
            else:
                # Each vector is folded in and not kept, exactly as a contribution would be.
                totals[accent].add(windows[:MAX_TAKES_PER_CONTRIBUTOR * 3])

    classifier = build_classifier(totals)
    results = {
        "method": "linear discriminant built only from per-accent counts, sums and sums of outer products",
        "features": FEATURES,
        "kilobytes_kept_per_region": round((FEATURES * FEATURES + FEATURES + 1) * 8 / 1024),
        "train_vectors": {CLASS_NAMES[a]: totals[a].count for a in CLASSES},
    }
    for name, (vectors, labels) in tests.items():
        guesses = classifier.classify(np.concatenate(vectors))
        recalls = [
            np.mean([g == accent for g, label in zip(guesses, labels) if label == accent]) for accent in CLASSES
        ]
        results[f"test_accuracy_{name}"] = round(float(np.mean(recalls)), 3)
    print(json.dumps(results, indent=2))

    report = json.loads(REPORT_PATH.read_text())
    report["region_totals"] = results
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")


def check_take(take, feature_set: str) -> tuple[dict | None, str | None]:
    """One exported take, checked and reduced to what learning needs, or the reason it was refused."""
    if not isinstance(take, dict):
        return None, "not a take"
    take_id, contributor, country = take.get("id"), take.get("contributor"), take.get("country")
    if not isinstance(take_id, str) or not UUID.match(take_id):
        return None, "bad take id"
    if not isinstance(contributor, str) or not UUID.match(contributor):
        return None, "bad contributor id"
    if not isinstance(country, str) or not COUNTRY.match(country):
        return None, "bad country"
    country = country.upper()
    region = take.get("region")
    if region is not None and (not isinstance(region, str) or region not in REGIONS.get(country, {})):
        return None, "bad region"

    natural, matches, accent = take.get("natural_voice"), take.get("matches_home_accent"), take.get("target_accent")
    if not isinstance(natural, bool) or not isinstance(matches, bool):
        return None, "bad labels"
    if (accent is not None) if natural else (accent not in ACCENTS):
        return None, "bad target accent"
    # The label is recomputed rather than trusted, so stored labels stay truthful.
    if matches != (not natural and HOME_ACCENT.get(country) == accent):
        return None, "labels disagree"

    if take.get("model_version") != feature_set:
        return None, "made with a different model version"
    features = take.get("features")
    if not isinstance(features, list) or len(features) != FEATURES:
        return None, "wrong feature length"
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in features):
        return None, "non-finite feature"
    if any(abs(v) > MAX_ABS_FEATURE for v in features):
        return None, "feature out of range"

    return {
        "id": take_id.lower(), "contributor": contributor.lower(), "country": country, "region": region,
        "target_accent": accent, "natural_voice": natural, "matches_home_accent": matches, "features": features,
    }, None


def route(take: dict) -> tuple[str, list[str]]:
    """Which totals a take joins: ("native", [country, ...]) or ("attempt", ["country>accent"]).

    Only a take in the speaker's normal voice says how a place sounds. Every practice
    take is someone aiming at a target accent, so it is an attempt, even when the
    target is the standard accent of the speaker's own country: a Scot practising
    standard British, or an American practising General American, is still imitating
    a standard.

      normal voice         -> native: country, and country-region when one was given
      any practice take    -> attempt: country>target accent
    """
    if take["natural_voice"]:
        keys = [take["country"]]
        if take["region"]:
            keys.append(f"{take['country']}-{take['region']}")
        return "native", keys
    return "attempt", [f"{take['country']}>{take['target_accent']}"]


def read_export(path: Path) -> tuple[list, str | None]:
    """The takes in one file exported from the app's "Your data" page, or a reason the file was skipped."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return [], "invalid json"
    if (
        not isinstance(document, dict) or document.get("app") != "phonemepro"
        or document.get("kind") != "contributions" or document.get("version") != EXPORT_VERSION
        or not isinstance(document.get("takes"), list)
    ):
        return [], "unrecognised file"
    return document["takes"], None


def ingest(source: Path, native_path: Path, attempts_path: Path, seen_path: Path, feature_set: str,
           min_contributors: int = MIN_CONTRIBUTORS, min_takes: int = MIN_TAKES, dry_run: bool = False,
           rebuild: bool = False) -> dict:
    """Fold every new, valid take in `source` (an export file or a folder of them) into the totals.

    With `dry_run` everything is checked and summarised as it would be, and nothing is written.
    With `rebuild` the existing totals and the list of counted takes are ignored and replaced, so
    the result reflects exactly the files in `source`: removing a file removes its takes.
    """
    stores = {}
    for kind, path in (("native", native_path), ("attempt", attempts_path)):
        totals, stored_set = load_totals(path) if path.exists() and not rebuild else ({}, feature_set)
        if stored_set != feature_set:
            raise SystemExit(f"{path} was built with model {stored_set}; start a new totals file for {feature_set}.")
        stores[kind] = totals
    # Take ids already counted, hashed. They are random and say nothing about the person.
    seen = set(seen_path.read_text().split()) if seen_path.exists() and not rebuild else set()

    skipped: dict[str, int] = {}
    refused: dict[str, int] = {}
    added: dict[tuple[str, str], int] = {}
    contributors: dict[str, set] = {}
    per_contributor: dict[str, int] = {}
    duplicates = over_cap = takes_added = 0

    for path in [source] if source.is_file() else sorted(source.rglob("*.json")):
        raw_takes, reason = read_export(path)
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        for raw in raw_takes:
            take, reason = check_take(raw, feature_set)
            if reason:
                refused[reason] = refused.get(reason, 0) + 1
                continue
            digest = hashlib.sha256(take["id"].encode()).hexdigest()[:24]
            if digest in seen:
                duplicates += 1
                continue
            if per_contributor.get(take["contributor"], 0) >= MAX_TAKES_PER_CONTRIBUTOR:
                over_cap += 1
                continue
            seen.add(digest)
            per_contributor[take["contributor"]] = per_contributor.get(take["contributor"], 0) + 1
            takes_added += 1
            kind, keys = route(take)
            for key in keys:
                stores[kind].setdefault(key, Totals.empty()).add(take["features"])
                added[kind, key] = added.get((kind, key), 0) + 1
            contributors.setdefault(take["country"], set()).add(take["contributor"])

    if not dry_run:
        for kind, path in (("native", native_path), ("attempt", attempts_path)):
            path.parent.mkdir(parents=True, exist_ok=True)
            save_totals(path, stores[kind], feature_set)
        seen_path.write_text("\n".join(sorted(seen)) + ("\n" if seen else ""))

    native, attempts = stores["native"], stores["attempt"]
    # Native totals are keyed by country ("GB") and by region within it ("GB-SCT").
    countries = sorted({key.split("-")[0] for key in native} | {key.split(">")[0] for key in attempts})
    return {
        "feature_set": feature_set,
        "dry_run": dry_run,
        "rebuilt": rebuild,
        "added": takes_added,
        "added_native": sum(count for (kind, key), count in added.items() if kind == "native" and "-" not in key),
        "added_attempts": sum(count for (kind, _), count in added.items() if kind == "attempt"),
        "duplicates": duplicates,
        "refused": refused,
        "over_contributor_cap": over_cap,
        "files_skipped": skipped,
        "thresholds": {"min_contributors": min_contributors, "min_takes": min_takes},
        "countries": {
            country: {
                "native_total": native[country].count if country in native else 0,
                "native_added": added.get(("native", country), 0),
                "regions": {
                    key.split("-")[1]: {"total": totals.count, "added": added.get(("native", key), 0)}
                    for key, totals in sorted(native.items()) if key.startswith(f"{country}-")
                },
                "attempts": {
                    key.split(">")[1]: {"total": totals.count, "added": added.get(("attempt", key), 0)}
                    for key, totals in sorted(attempts.items()) if key.split(">")[0] == country
                },
                "contributors_this_run": len(contributors.get(country, ())),
                # Contributors are not remembered between runs, so readiness is judged on this run's files.
                "ready": (
                    country in native and native[country].count >= min_takes
                    and len(contributors.get(country, ())) >= min_contributors
                ),
            }
            for country in countries
        },
    }


def build_reference(path: Path = REFERENCE_PATH) -> dict:
    """Totals for native speakers of each practice accent, from VCTK one-second windows."""
    from scripts.accent_data import ACCENT_SPEAKERS, PROBE_LAYERS
    from scripts.train_accent_head import load_stats

    layer_index = PROBE_LAYERS.index(int(np.load(PROJECTION_PATH)["layer"]))
    totals = {}
    for accent, group in REFERENCE_ACCENTS.items():
        totals[accent] = Totals.empty()
        for speaker in ACCENT_SPEAKERS[group]:
            stats = load_stats(speaker, layer_index)
            totals[accent].add(project(stats[:, 1:].reshape(-1, stats.shape[-1])))
    feature_set = json.loads(MODEL_META_PATH.read_text())["version"]
    path.parent.mkdir(parents=True, exist_ok=True)
    save_totals(path, totals, feature_set)
    return {accent: t.count for accent, t in totals.items()}


def report(native: dict, attempts: dict, reference: dict, contributors: dict | None = None,
           min_contributors: int = MIN_CONTRIBUTORS, min_takes: int = MIN_TAKES) -> dict:
    """What the totals can tell so far.

    native       which places have enough normal-voice takes, and whether those that do can be
                 told apart (a classifier needs at least two)
    attempts     for each "country attempting accent" group, its average squared distance to
                 the native reference accents: lower is closer. `closest` is the reference it
                 sits nearest on average, which should be the target if attempts are landing.

    `contributors` maps a country to how many people contributed in the last ingest; without it
    readiness is judged on takes alone and marked as unknown.
    """
    contributors = contributors or {}
    places = {}
    for key, totals in sorted(native.items()):
        country = key.split("-")[0]
        people = contributors.get(country)
        enough_people = None if people is None or "-" in key else people >= min_contributors
        places[key] = {
            "takes": totals.count,
            "contributors": people if "-" not in key else None,
            "enough_takes": totals.count >= min_takes,
            "enough_contributors": enough_people,
            "ready": totals.count >= min_takes and enough_people is not False,
        }
    ready = {key: native[key] for key, place in places.items() if place["ready"]}
    classifier = build_classifier(ready)

    distances = {}
    centres = build_classifier(reference) if len(reference) >= 2 else None
    if centres is not None:
        for key, totals in sorted({**attempts, **native}.items()):
            target = key.split(">")[1] if ">" in key else None
            by_accent = {
                accent: round(mean_squared_distance(totals, mean, centres.precision), 2)
                for accent, mean in zip(centres.regions, centres.means)
            }
            distances[key] = {
                "takes": totals.count,
                "kind": "attempt" if target else "native",
                "target": target,
                "distance_to": by_accent,
                "closest": min(by_accent, key=by_accent.get),
                "lands_on_target": None if target not in by_accent else min(by_accent, key=by_accent.get) == target,
            }
    return {
        "thresholds": {"min_contributors": min_contributors, "min_takes": min_takes},
        "places": places,
        "can_tell_apart": classifier.regions if classifier else [],
        "reference_accents": sorted(reference),
        "distances": distances,
    }


def print_report(result: dict) -> None:
    thresholds = result["thresholds"]
    print(f"Normal-voice takes by place (ready at {thresholds['min_takes']} takes from "
          f"{thresholds['min_contributors']} people):")
    for key, place in result["places"].items():
        people = "?" if place["contributors"] is None else place["contributors"]
        print(f"  {key:10} {place['takes']:5} takes  {people:>3} people  {'ready' if place['ready'] else 'not yet'}")
    if not result["places"]:
        print("  none yet")
    apart = result["can_tell_apart"]
    print(f"Places that can be told apart now: {', '.join(apart) if apart else 'none (needs two that are ready)'}")

    if not result["reference_accents"]:
        print("\nNo native reference accents. Run: python -m scripts.learn_regions reference")
        return
    names = result["reference_accents"]
    print(f"\nAverage squared distance to native reference accents ({', '.join(names)}); lower is closer:")
    for key, entry in result["distances"].items():
        cells = "  ".join(f"{name} {entry['distance_to'][name]:8.2f}" for name in names)
        verdict = "" if entry["lands_on_target"] is None else ("  on target" if entry["lands_on_target"] else "  closer to another accent")
        print(f"  {key:10} {entry['takes']:5} takes  {cells}  closest {entry['closest']}{verdict}")
    if not result["distances"]:
        print("  no takes yet")
    print("With few takes these averages mostly reflect which words were said and who said them.")


def print_summary(summary: dict) -> None:
    if summary["dry_run"]:
        print("Dry run: nothing was written. This is what an ingest would do.")
    print(f"Model {summary['feature_set']}: added {summary['added']} takes "
          f"({summary['added_native']} native accent, {summary['added_attempts']} attempts), "
          f"{summary['duplicates']} already counted, {summary['over_contributor_cap']} over the per-contributor cap.")
    for label, reasons in (("Refused takes", summary["refused"]), ("Skipped files", summary["files_skipped"])):
        for reason, count in sorted(reasons.items()):
            print(f"  {label}: {count} x {reason}")
    print(f"\n{'country':8} {'native':>7} {'(new)':>6}  {'contributors':>12}  {'ready':5}  attempts at other accents, total (new)")
    for country, entry in summary["countries"].items():
        attempts = ", ".join(f"{accent} {a['total']} ({a['added']})" for accent, a in entry["attempts"].items()) or "none"
        print(f"{country:8} {entry['native_total']:7} {entry['native_added']:6}  {entry['contributors_this_run']:12}  "
              f"{'yes' if entry['ready'] else 'no':5}  {attempts}")
        for region, r in entry["regions"].items():
            print(f"  {region:6} {r['total']:7} {r['added']:6}  {REGIONS[country][region]}")
    thresholds = summary["thresholds"]
    print(f"\nA country is ready at {thresholds['min_takes']} native takes from "
          f"{thresholds['min_contributors']} contributors in one run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fit-projection")
    commands.add_parser("simulate")
    ingest_parser = commands.add_parser("ingest")
    ingest_parser.add_argument("source", type=Path, help="an export file, or a folder of them")
    ingest_parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    ingest_parser.add_argument("--dry-run", action="store_true", help="check and summarise, write nothing")
    ingest_parser.add_argument("--rebuild", action="store_true",
                               help="replace the totals with exactly what these files hold (pilot)")
    commands.add_parser("reference")
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--json", action="store_true", help="print the report as JSON")
    ingest_parser.add_argument("--min-contributors", type=int, default=MIN_CONTRIBUTORS)
    ingest_parser.add_argument("--min-takes", type=int, default=MIN_TAKES)
    args = parser.parse_args()

    if args.command == "fit-projection":
        fit_projection()
    elif args.command == "simulate":
        simulate()
    elif args.command == "reference":
        counts = build_reference()
        print(f"Wrote {REFERENCE_PATH}: " + ", ".join(f"{accent} {count} windows" for accent, count in counts.items()))
    elif args.command == "report":
        def load(path):
            return load_totals(path)[0] if path.exists() else {}
        summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {"countries": {}}
        contributors = {country: entry["contributors_this_run"] for country, entry in summary["countries"].items()}
        result = report(load(TOTALS_PATH), load(ATTEMPTS_PATH), load(REFERENCE_PATH), contributors)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if summary.get("rebuilt") is False:
                print("Note: people are counted from the last ingest only. Use ingest --rebuild during a pilot.\n")
            print_report(result)
    else:
        if not args.source.exists():
            raise SystemExit(f"{args.source} does not exist.")
        feature_set = json.loads(MODEL_META_PATH.read_text())["version"]
        summary = ingest(args.source, TOTALS_PATH, ATTEMPTS_PATH, SEEN_PATH, feature_set,
                         args.min_contributors, args.min_takes, args.dry_run, args.rebuild)
        if not args.dry_run:
            SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print_summary(summary)


if __name__ == "__main__":
    main()
