"""
The shipped classifier must keep predicting what it predicts today.

The 241 vectors in data/tap_dataset are the only real recorded input this
project has. They pin the classifier's arithmetic so a refactor cannot drift
it silently. They do NOT pin the tap state machine: the collector records the
finished feature vector after the machine has already decided a press ended,
so this data is the machine's output and cannot be replayed as its input.
"""

import json
from pathlib import Path

import pytest

from src.tap_classifier.tap_classifier import TapClassifier

BASELINE = Path(__file__).parent / 'data' / 'tap_predictions_baseline.json'
MODEL = Path(__file__).parents[1] / 'models' / 'tap_model.json'


@pytest.fixture(scope='module')
def baseline():
    with BASELINE.open(encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(scope='module')
def classifier():
    clf = TapClassifier()
    clf.load_model(str(MODEL))
    return clf


def test_baseline_covers_every_collected_sample(baseline):
    assert len(baseline) == 241


def test_predictions_match_the_baseline(classifier, baseline):
    drifted = []
    for i, row in enumerate(baseline):
        got = float(classifier.predict(row['features']))
        if abs(got - row['probability']) > 1e-9:
            drifted.append((i, row['probability'], got))

    assert not drifted, f'{len(drifted)} prediction(s) drifted, first: {drifted[:3]}'


def test_probabilities_stay_in_range(classifier, baseline):
    for row in baseline:
        prob = float(classifier.predict(row['features']))
        assert 0.0 <= prob <= 1.0
