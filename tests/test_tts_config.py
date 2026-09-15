"""TTSConfig carries the settings both the generator and the runtime fallback read."""

from src.config import TTSConfig


def test_voice_and_paths_are_configured():
    assert TTSConfig.VOICE == 'uk_UA-ukrainian_tts-medium'
    assert TTSConfig.VOICES_DIR == 'models/tts_voices'
    assert TTSConfig.GENERATED_SUBDIR == 'tts'


def test_runtime_fallback_is_on_by_default():
    assert TTSConfig.RUNTIME_FALLBACK is True


def test_piper_bin_defaults_to_path_lookup():
    assert TTSConfig.PIPER_BIN is None
