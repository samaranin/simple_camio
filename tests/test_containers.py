"""The two containers that replaced untyped dicts."""

import dataclasses

import pytest

from src.core.containers import Components, Workers


def _components(**overrides):
    fields = {name: object() for name in (
        'model', 'cam_port', 'model_detector', 'pose_detector',
        'gesture_detector', 'motion_filter', 'interact', 'camio_player',
        'crickets_player', 'heartbeat_player',
    )}
    fields.update(overrides)
    return Components(**fields)


def test_components_exposes_every_field():
    c = _components()

    for name in ('model', 'cam_port', 'model_detector', 'pose_detector',
                 'gesture_detector', 'motion_filter', 'interact',
                 'camio_player', 'crickets_player', 'heartbeat_player'):
        assert getattr(c, name) is not None


def test_components_rejects_an_unknown_field():
    """A mistyped name fails at construction, not at some far-away use."""
    with pytest.raises(TypeError):
        _components(camio_playr=object())


def test_components_rejects_a_missing_field():
    with pytest.raises(TypeError):
        Components(model=object())


def test_components_is_frozen():
    c = _components()

    with pytest.raises(dataclasses.FrozenInstanceError):
        c.model = object()


def test_workers_exposes_every_field():
    w = Workers(
        audio_worker=object(), pose_worker=object(), sift_worker=object(),
        pose_queue=object(), sift_queue=object(), lock=object(),
    )

    assert w.audio_worker is not None
    assert w.lock is not None


def test_workers_is_frozen():
    w = Workers(
        audio_worker=object(), pose_worker=object(), sift_worker=object(),
        pose_queue=object(), sift_queue=object(), lock=object(),
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        w.lock = object()
