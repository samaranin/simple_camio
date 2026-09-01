"""Zone mapping and the mode filter that smooths it."""

import numpy as np
import pytest

from src.config import InteractionConfig
from src.core.interaction_policy import InteractionPolicy2D


@pytest.fixture
def policy(zone_map_model):
    return InteractionPolicy2D(zone_map_model)


def _touch(x, y):
    """A position on the map, close enough in z to count as touching."""
    return np.array([x, y, 0.0])


def test_filter_starts_empty(policy):
    assert (policy.zone_filter == -1).all()


def test_repeated_touch_settles_on_one_zone(policy):
    """Fill the ring buffer with one zone; the mode has to be that zone."""
    zone = None
    for _ in range(policy.ZONE_FILTER_SIZE):
        zone = policy.push_gesture(_touch(15, 15))

    assert zone >= 0
    settled = zone

    # Another sample of the same zone cannot change the answer.
    assert policy.push_gesture(_touch(15, 15)) == settled


def test_distinct_colours_map_to_distinct_zones(policy):
    for _ in range(policy.ZONE_FILTER_SIZE):
        top = policy.push_gesture(_touch(15, 15))
    policy.reset_zone_filter()
    for _ in range(policy.ZONE_FILTER_SIZE):
        middle = policy.push_gesture(_touch(15, 45))

    assert top != middle


def test_single_outlier_does_not_flip_the_zone(policy):
    """The mode filter exists to absorb exactly this."""
    for _ in range(policy.ZONE_FILTER_SIZE):
        settled = policy.push_gesture(_touch(15, 15))

    assert policy.push_gesture(_touch(15, 45)) == settled


def test_hand_too_high_reports_no_zone(policy):
    """Beyond the z threshold the user is not touching the map."""
    above = np.array([15, 15, InteractionConfig.Z_THRESHOLD + 10.0])

    assert policy.push_gesture(above) == -1


def test_reset_clears_the_buffer(policy):
    for _ in range(policy.ZONE_FILTER_SIZE):
        policy.push_gesture(_touch(15, 15))

    policy.reset_zone_filter()

    assert (policy.zone_filter == -1).all()
    assert policy.zone_filter_cnt == 0
