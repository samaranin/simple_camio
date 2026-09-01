"""Utility helpers used across the pipeline."""

import json

import numpy as np
import pytest

from src.core.utils import (
    color_to_index,
    is_gesture_valid,
    load_map_parameters,
    normalize_gesture_location,
)


class TestNormalizeGestureLocation:
    def test_none_stays_none(self):
        assert normalize_gesture_location(None) is None

    def test_flat_triple_is_returned_as_three_elements(self):
        out = normalize_gesture_location(np.array([1.0, 2.0, 3.0]))

        assert out is not None
        assert np.asarray(out).size == 3

    def test_nested_array_is_flattened_to_three(self):
        out = normalize_gesture_location(np.array([[1.0, 2.0, 3.0]]))

        assert np.asarray(out).size == 3


class TestIsGestureValid:
    def test_none_is_not_valid(self):
        assert is_gesture_valid(None) is False

    def test_a_real_position_is_valid(self):
        assert is_gesture_valid(np.array([1.0, 2.0, 3.0])) is True


class TestColorToIndex:
    def test_distinct_colours_give_distinct_indices(self):
        assert color_to_index((255, 0, 0)) != color_to_index((0, 255, 0))

    def test_same_colour_gives_the_same_index(self):
        assert color_to_index((12, 34, 56)) == color_to_index((12, 34, 56))


class TestLoadMapParameters:
    def test_reads_the_model_block(self, tmp_path):
        path = tmp_path / 'model.json'
        path.write_text(json.dumps({
            'model': {'modelType': 'sift_2d_mediapipe', 'name': 'Test'}
        }), encoding='utf-8')

        model = load_map_parameters(str(path))

        assert model['modelType'] == 'sift_2d_mediapipe'
        assert model['name'] == 'Test'

    def test_missing_file_raises(self, tmp_path):
        # load_map_parameters never lets a missing file propagate as
        # FileNotFoundError/OSError: it logs and calls sys.exit(1) so a
        # daemon run fails loudly instead of hanging. That raises
        # SystemExit, which is what this pins.
        with pytest.raises(SystemExit):
            load_map_parameters(str(tmp_path / 'nope.json'))
