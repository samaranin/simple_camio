"""
_load_sound is the single place that turns a path into a playable sound, so it is
also the single place that has to survive a missing file.
"""

import src.audio.audio as audio_module
from src.audio.audio import ZoneAudioPlayer


def _player_without_init():
    """A ZoneAudioPlayer instance with no __init__ run - we only want the method."""
    return ZoneAudioPlayer.__new__(ZoneAudioPlayer)


def test_missing_file_returns_none_and_does_not_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_module, 'USE_PYGLET', False)
    monkeypatch.setattr(audio_module, 'USE_PYGAME', False)
    assert _player_without_init()._load_sound(str(tmp_path / 'nope.wav')) is None


def test_empty_path_returns_none(monkeypatch):
    monkeypatch.setattr(audio_module, 'USE_PYGLET', False)
    monkeypatch.setattr(audio_module, 'USE_PYGAME', False)
    assert _player_without_init()._load_sound('') is None


def test_no_backend_returns_none_for_a_file_that_exists(monkeypatch, tmp_path):
    real = tmp_path / 'real.wav'
    real.write_bytes(b'RIFF')
    monkeypatch.setattr(audio_module, 'USE_PYGLET', False)
    monkeypatch.setattr(audio_module, 'USE_PYGAME', False)
    assert _player_without_init()._load_sound(str(real)) is None


def test_a_backend_that_raises_is_contained(monkeypatch, tmp_path):
    """A corrupt clip must not take the application down."""
    real = tmp_path / 'real.wav'
    real.write_bytes(b'not really a wav')

    class Boom:
        @staticmethod
        def load(path, streaming=False):
            raise RuntimeError('decoder blew up')

    monkeypatch.setattr(audio_module, 'USE_PYGLET', True)
    monkeypatch.setattr(audio_module, 'USE_PYGAME', False)
    monkeypatch.setitem(__import__('sys').modules, 'pyglet.media', Boom)
    assert _player_without_init()._load_sound(str(real)) is None
