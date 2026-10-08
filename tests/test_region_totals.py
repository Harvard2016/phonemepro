import numpy as np
import pytest

from pronunciation.region_totals import Totals, build_classifier, load_totals, save_totals

FEATURES = 12


def cloud(rng, centre, count):
    return centre + rng.standard_normal((count, FEATURES))


@pytest.fixture
def rng():
    return np.random.default_rng(4)


def test_totals_do_not_depend_on_how_vectors_arrive(rng):
    vectors = rng.standard_normal((40, FEATURES))
    at_once, one_by_one, merged = Totals.empty(FEATURES), Totals.empty(FEATURES), Totals.empty(FEATURES)
    at_once.add(vectors)
    for vector in vectors:
        one_by_one.add(vector)
    first, second = Totals.empty(FEATURES), Totals.empty(FEATURES)
    first.add(vectors[:15])
    second.add(vectors[15:])
    merged.merge(first)
    merged.merge(second)

    for other in (one_by_one, merged):
        assert other.count == at_once.count == 40
        assert np.allclose(other.total, at_once.total)
        assert np.allclose(other.outer, at_once.outer)
    assert np.allclose(at_once.mean, vectors.mean(axis=0))


def test_storage_is_fixed_however_many_vectors_are_added(rng):
    totals = Totals.empty(FEATURES)
    size_before = totals.total.nbytes + totals.outer.nbytes
    totals.add(rng.standard_normal((5000, FEATURES)))
    assert totals.total.nbytes + totals.outer.nbytes == size_before


def test_classifier_from_totals_separates_regions(rng):
    centres = {"GB": np.zeros(FEATURES), "US": np.full(FEATURES, 1.5), "IN": np.r_[np.full(6, -2.0), np.zeros(6)]}
    totals = {}
    for region, centre in centres.items():
        totals[region] = Totals.empty(FEATURES)
        totals[region].add(cloud(rng, centre, 300))

    classifier = build_classifier(totals)
    assert classifier.regions == ["GB", "IN", "US"]
    for region, centre in centres.items():
        guesses = classifier.classify(cloud(rng, centre, 200))
        assert guesses.count(region) / len(guesses) > 0.9
    assert classifier.scores(np.zeros(FEATURES)).shape == (1, 3)


def test_classifier_matches_one_fitted_on_the_raw_vectors(rng):
    a, b = cloud(rng, np.zeros(FEATURES), 150), cloud(rng, np.ones(FEATURES), 250)
    totals = {"A": Totals.empty(FEATURES), "B": Totals.empty(FEATURES)}
    totals["A"].add(a)
    totals["B"].add(b)
    classifier = build_classifier(totals)

    assert np.allclose(classifier.means, [a.mean(axis=0), b.mean(axis=0)])
    centred = np.concatenate([a - a.mean(axis=0), b - b.mean(axis=0)])
    covariance = centred.T @ centred / (len(centred) - 2)
    covariance += 1e-3 * np.trace(covariance) / FEATURES * np.eye(FEATURES)
    assert np.allclose(classifier.precision, np.linalg.inv(covariance))


def test_small_regions_are_left_out_and_one_region_is_not_enough(rng):
    totals = {name: Totals.empty(FEATURES) for name in ("A", "B", "C")}
    totals["A"].add(cloud(rng, np.zeros(FEATURES), 100))
    totals["B"].add(cloud(rng, np.ones(FEATURES), 100))
    totals["C"].add(cloud(rng, np.ones(FEATURES), 5))
    assert build_classifier(totals, min_count=50).regions == ["A", "B"]
    assert build_classifier(totals, min_count=500) is None
    assert build_classifier({"A": totals["A"]}) is None


def test_save_and_load_round_trip(rng, tmp_path):
    totals = {"GB": Totals.empty(FEATURES), "US": Totals.empty(FEATURES)}
    totals["GB"].add(cloud(rng, np.zeros(FEATURES), 20))
    totals["US"].add(cloud(rng, np.ones(FEATURES), 30))
    path = tmp_path / "totals.npz"
    save_totals(path, totals, "abc123")

    loaded, feature_set = load_totals(path)
    assert feature_set == "abc123"
    assert set(loaded) == {"GB", "US"}
    assert loaded["US"].count == 30
    assert np.allclose(loaded["GB"].outer, totals["GB"].outer)
