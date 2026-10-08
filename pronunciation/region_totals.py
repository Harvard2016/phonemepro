"""Learn what each region sounds like from running totals, without keeping anyone's data.

A contributed take arrives as a short feature vector (see scripts/learn_regions.py
for where it comes from). For each region only three things are kept: how many
vectors were added, their sum, and the sum of their outer products. A vector is
folded into those totals and then dropped, so storage per region is fixed no
matter how many people contribute, and no single take can be read back out once
others have been added.

Those totals are exactly what a linear discriminant needs: each region's mean and
a shared covariance. On VCTK this matches a classifier trained on the individual
recordings.
"""
from dataclasses import dataclass

import numpy as np

FEATURES = 256
# Shrink the covariance towards a multiple of the identity so it can be inverted
# while a region still has few contributions.
SHRINKAGE = 1e-3


@dataclass
class Totals:
    count: int
    total: np.ndarray  # [FEATURES]
    outer: np.ndarray  # [FEATURES, FEATURES]

    @classmethod
    def empty(cls, features: int = FEATURES) -> "Totals":
        return cls(0, np.zeros(features), np.zeros((features, features)))

    def add(self, vectors: np.ndarray) -> None:
        """Fold one vector or a batch of vectors into the totals."""
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
        self.count += len(vectors)
        self.total += vectors.sum(axis=0)
        self.outer += vectors.T @ vectors

    def merge(self, other: "Totals") -> None:
        self.count += other.count
        self.total += other.total
        self.outer += other.outer

    @property
    def mean(self) -> np.ndarray:
        return self.total / max(self.count, 1)


@dataclass
class RegionClassifier:
    regions: list[str]
    means: np.ndarray  # [regions, FEATURES]
    precision: np.ndarray  # inverse of the shared covariance

    def scores(self, vectors: np.ndarray) -> np.ndarray:
        """Negative squared distance to each region's mean; higher is closer."""
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
        differences = vectors[:, None, :] - self.means[None]
        return -np.einsum("nrf,fg,nrg->nr", differences, self.precision, differences)

    def classify(self, vectors: np.ndarray) -> list[str]:
        return [self.regions[i] for i in self.scores(vectors).argmax(axis=1)]


def build_classifier(totals: dict[str, Totals], min_count: int = 1) -> RegionClassifier | None:
    """Linear discriminant from totals alone. Regions with fewer than `min_count` vectors are left out."""
    regions = sorted(region for region, t in totals.items() if t.count >= max(min_count, 1))
    if len(regions) < 2:
        return None

    means = np.stack([totals[region].mean for region in regions])
    count = sum(totals[region].count for region in regions)
    # Within-region scatter: sum of outer products minus each region's own mean term.
    scatter = sum(
        totals[region].outer - totals[region].count * np.outer(mean, mean)
        for region, mean in zip(regions, means)
    )
    covariance = scatter / max(count - len(regions), 1)
    size = covariance.shape[0]
    covariance = covariance + SHRINKAGE * (np.trace(covariance) / size) * np.eye(size)
    return RegionClassifier(regions, means, np.linalg.inv(covariance))


def mean_squared_distance(totals: Totals, centre: np.ndarray, precision: np.ndarray) -> float:
    """Average squared distance of the vectors in `totals` from `centre`, without the vectors themselves.

    E[(x - m)' P (x - m)] expands to terms in the count, the sum and the sum of outer products,
    which is all that is kept. Lower means the group sits closer to `centre`.
    """
    if totals.count == 0:
        return float("nan")
    spread = np.sum(precision * totals.outer)  # trace(P @ outer), P symmetric
    cross = centre @ precision @ totals.total
    return float((spread - 2 * cross) / totals.count + centre @ precision @ centre)


def save_totals(path, totals: dict[str, Totals], feature_set: str) -> None:
    regions = sorted(totals)
    np.savez_compressed(
        path,
        feature_set=feature_set,
        regions=np.array(regions),
        counts=np.array([totals[r].count for r in regions]),
        totals=np.stack([totals[r].total for r in regions]) if regions else np.zeros((0, FEATURES)),
        outers=np.stack([totals[r].outer for r in regions]) if regions else np.zeros((0, FEATURES, FEATURES)),
    )


def load_totals(path) -> tuple[dict[str, Totals], str]:
    data = np.load(path)
    totals = {
        str(region): Totals(int(count), total.copy(), outer.copy())
        for region, count, total, outer in zip(data["regions"], data["counts"], data["totals"], data["outers"])
    }
    return totals, str(data["feature_set"])
