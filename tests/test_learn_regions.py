"""Ingesting takes exported from the app's "Your data" page."""
import copy
import json
import uuid
from pathlib import Path

import numpy as np
import pytest

from pronunciation.region_totals import FEATURES, load_totals
from scripts.learn_regions import (
    HOME_ACCENT,
    MAX_ABS_FEATURE,
    MAX_TAKES_PER_CONTRIBUTOR,
    check_take,
    ingest,
    print_summary,
    route,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "contribution_export.json").read_text())
MODEL = "fixture-model"


def make_take(**changes) -> dict:
    """A valid take: someone from India practising American."""
    take = copy.deepcopy(FIXTURE["export"]["takes"][3])
    take["id"] = str(uuid.uuid4())
    take.update(changes)
    return take


def write_export(folder: Path, takes: list, name: str = "export.json", **changes) -> Path:
    document = {**FIXTURE["export"], "takes": takes, **changes}
    path = folder / name
    path.write_text(json.dumps(document))
    return path


@pytest.fixture
def run(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()

    def ingest_here(takes=None, **options):
        if takes is not None:
            write_export(inbox, takes, f"{uuid.uuid4()}.json")
        return ingest(inbox, tmp_path / "native.npz", tmp_path / "attempts.npz", tmp_path / "seen.txt", MODEL, **options)

    ingest_here.inbox = inbox
    ingest_here.root = tmp_path
    return ingest_here


def test_the_shared_fixture_is_accepted_and_routed_as_the_app_expects(run):
    assert HOME_ACCENT == FIXTURE["home_accent"]
    for raw in FIXTURE["export"]["takes"]:
        take, reason = check_take(raw, MODEL)
        assert reason is None
        assert list(route(take)) == FIXTURE["routes"][raw["id"]]

    summary = run(FIXTURE["export"]["takes"])
    assert (summary["added"], summary["added_native"], summary["added_attempts"]) == (4, 2, 2)
    assert summary["refused"] == {} and summary["files_skipped"] == {}


def test_imitating_an_accent_never_counts_as_the_home_accent(run):
    contributor = str(uuid.uuid4())
    takes = [
        make_take(contributor=contributor, country="GB", target_accent="rp", matches_home_accent=True),
        make_take(contributor=contributor, country="GB", target_accent="ga"),
        make_take(contributor=contributor, country="GB", target_accent="au"),
        make_take(contributor=contributor, country="GB", target_accent=None, natural_voice=True, score=None),
        make_take(contributor=contributor, country="IN", target_accent="rp"),
        make_take(contributor=contributor, country="IN", target_accent=None, natural_voice=True, score=None),
    ]
    summary = run(takes)

    native, _ = load_totals(run.root / "native.npz")
    attempts, _ = load_totals(run.root / "attempts.npz")
    assert {region: totals.count for region, totals in native.items()} == {"GB": 2, "IN": 1}
    assert {key: totals.count for key, totals in attempts.items()} == {"GB>ga": 1, "GB>au": 1, "IN>rp": 1}
    assert np.allclose(native["IN"].total, takes[5]["features"])

    assert summary["countries"]["GB"] == {
        "native_total": 2, "native_added": 2,
        "attempts": {"au": {"total": 1, "added": 1}, "ga": {"total": 1, "added": 1}},
        "contributors_this_run": 1, "ready": False,
    }
    assert summary["countries"]["IN"]["attempts"] == {"rp": {"total": 1, "added": 1}}


@pytest.mark.parametrize("changes, reason", [
    ({"id": "not-a-uuid"}, "bad take id"),
    ({"id": 7}, "bad take id"),
    ({"contributor": "someone@example.com"}, "bad contributor id"),
    ({"contributor": None}, "bad contributor id"),
    ({"country": "India"}, "bad country"),
    ({"country": "1N"}, "bad country"),
    ({"country": None}, "bad country"),
    ({"region": "x" * 61}, "bad region"),
    ({"region": 5}, "bad region"),
    ({"natural_voice": "yes"}, "bad labels"),
    ({"matches_home_accent": None}, "bad labels"),
    ({"target_accent": "scouse"}, "bad target accent"),
    ({"target_accent": None}, "bad target accent"),
    ({"natural_voice": True}, "bad target accent"),  # a normal-voice take has no target
    ({"matches_home_accent": True}, "labels disagree"),  # India practising American is not a home accent
    ({"country": "US", "matches_home_accent": False}, "labels disagree"),
    ({"model_version": "older-model"}, "made with a different model version"),
    ({"features": [0.0] * (FEATURES - 1)}, "wrong feature length"),
    ({"features": [0.0] * (FEATURES + 1)}, "wrong feature length"),
    ({"features": "0.1 0.2"}, "wrong feature length"),
    ({"features": None}, "wrong feature length"),
    ({"features": [float("nan")] + [0.0] * (FEATURES - 1)}, "non-finite feature"),
    ({"features": [float("inf")] + [0.0] * (FEATURES - 1)}, "non-finite feature"),
    ({"features": ["0.5"] + [0.0] * (FEATURES - 1)}, "non-finite feature"),
    ({"features": [True] + [0.0] * (FEATURES - 1)}, "non-finite feature"),
    ({"features": [MAX_ABS_FEATURE + 1] + [0.0] * (FEATURES - 1)}, "feature out of range"),
])
def test_malformed_takes_are_refused_with_a_reason(run, changes, reason):
    assert check_take(make_take(**changes), MODEL) == (None, reason)

    good = make_take()
    summary = run([make_take(**changes), good])
    assert summary["refused"] == {reason: 1}
    assert summary["added"] == 1
    attempts, _ = load_totals(run.root / "attempts.npz")
    assert np.allclose(attempts["IN>ga"].total, good["features"])


def test_non_finite_values_written_as_bare_json_tokens_are_refused(run):
    # Python's json writes NaN and Infinity as bare tokens, and reads them back.
    text = json.dumps({**FIXTURE["export"], "takes": [make_take(features=[float("nan")] * FEATURES)]})
    assert "NaN" in text
    (run.inbox / "nan.json").write_text(text)
    summary = run()
    assert summary["refused"] == {"non-finite feature": 1}
    assert summary["added"] == 0


def test_things_that_are_not_takes_or_exports_are_skipped(run):
    (run.inbox / "broken.json").write_text("{not json")
    (run.inbox / "history.json").write_text(json.dumps({"app": "phonemepro", "version": 1, "attempts": []}))
    (run.inbox / "list.json").write_text("[1, 2, 3]")
    write_export(run.inbox, [make_take()], "old.json", version=1)
    write_export(run.inbox, "many", "takes-not-a-list.json")
    write_export(run.inbox, ["a string", 3, None, make_take()], "mixed.json")

    summary = run()
    assert summary["files_skipped"] == {"invalid json": 1, "unrecognised file": 4}
    assert summary["refused"] == {"not a take": 3}
    assert summary["added"] == 1


def test_a_take_is_counted_once_however_often_it_is_sent(run):
    takes = [make_take(), make_take()]
    assert run(takes + [takes[0]])["added"] == 2  # repeated inside one file

    again = run(takes)  # the same takes in a second file, on a later run
    assert again["added"] == 0
    assert again["duplicates"] == 2 + 2 + 1  # both files are read again
    attempts, _ = load_totals(run.root / "attempts.npz")
    assert attempts["IN>ga"].count == 2

    # Ids differing only in case are the same take.
    assert run([{**takes[0], "id": takes[0]["id"].upper()}])["added"] == 0


def test_one_contributor_cannot_outweigh_a_region(run):
    heavy, light = str(uuid.uuid4()), str(uuid.uuid4())
    takes = [make_take(contributor=heavy) for _ in range(MAX_TAKES_PER_CONTRIBUTOR + 7)]
    takes += [make_take(contributor=light) for _ in range(3)]
    summary = run(takes)
    assert summary["over_contributor_cap"] == 7
    assert summary["added"] == MAX_TAKES_PER_CONTRIBUTOR + 3
    assert summary["countries"]["IN"]["contributors_this_run"] == 2


def test_a_country_is_ready_only_with_enough_native_takes_and_people(run):
    people = [str(uuid.uuid4()) for _ in range(3)]
    native = [make_take(contributor=p, country="AU", target_accent="au", matches_home_accent=True) for p in people * 2]
    attempts = [make_take(contributor=p, country="NZ", target_accent="au") for p in people * 2]
    summary = run(native + attempts, min_contributors=3, min_takes=6)
    assert summary["countries"]["AU"]["ready"] is True
    assert summary["countries"]["NZ"]["ready"] is False  # attempts alone never make a country ready
    assert run([], min_contributors=4, min_takes=6)["countries"]["AU"]["ready"] is False


def test_totals_from_another_model_are_not_mixed(run):
    run([make_take()])
    with pytest.raises(SystemExit):
        ingest(run.inbox, run.root / "native.npz", run.root / "attempts.npz", run.root / "seen.txt", "newer-model")


def test_a_single_file_can_be_ingested_and_summarised(tmp_path, capsys):
    path = write_export(tmp_path, FIXTURE["export"]["takes"])
    summary = ingest(path, tmp_path / "n.npz", tmp_path / "a.npz", tmp_path / "seen.txt", MODEL)
    print_summary(summary)
    printed = capsys.readouterr().out
    assert "added 4 takes (2 native accent, 2 attempts)" in printed
    assert any(line.split()[:3] == ["GB", "2", "2"] and "ga 1 (1)" in line for line in printed.splitlines())
    # Only hashed ids are remembered, never the ids themselves.
    seen = (tmp_path / "seen.txt").read_text()
    assert len(seen.split()) == 4
    assert all(take["id"] not in seen for take in FIXTURE["export"]["takes"])
