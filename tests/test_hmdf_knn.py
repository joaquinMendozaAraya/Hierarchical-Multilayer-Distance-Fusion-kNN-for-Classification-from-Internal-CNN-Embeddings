from __future__ import annotations

import numpy as np

from models.hmdf_knn import HMDFKNN


def synthetic_views(seed: int = 7):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, 12))

    def split(samples_per_class: int):
        labels = np.repeat(np.arange(3), samples_per_class)
        base = centers[labels] + rng.normal(scale=0.6, size=(len(labels), 12))
        views = [
            base[:, :4] + rng.normal(scale=0.5, size=(len(labels), 4)),
            base[:, :8] + rng.normal(scale=0.25, size=(len(labels), 8)),
            base + rng.normal(scale=0.15, size=base.shape),
            rng.normal(size=(len(labels), 6)),
        ]
        return views, labels

    return split(20), split(8), split(8)


def test_selection_is_validation_only_and_reproducible():
    train, val, test = synthetic_views()
    first = HMDFKNN(seed=42).fit(train[0], train[1], val[0], val[1])
    second = HMDFKNN(seed=42).fit(train[0], train[1], val[0], val[1])

    assert first.selection_ == second.selection_
    assert first.selection_.candidate_count == 185
    assert first.selection_.k in {1, 3, 5, 7, 11}
    assert np.isclose(sum(first.selection_.weights), 1.0)
    metrics = first.evaluate(test[0], test[1])
    assert 0.0 <= metrics["f1_macro"] <= 1.0
