"""
The fallback runs once at load, so a finger resting on a zone never waits for
synthesis, and a broken TTS setup never takes the player down with it.
"""

import src.audio.audio as audio_module
from src.audio.audio import ZoneAudioPlayer
from src.tts import engine


def _player(model):
    player = ZoneAudioPlayer.__new__(ZoneAudioPlayer)
    player.model = model
    return player


def _model(tmp_path):
    return {
        'hotspots': [
            {'textDescription': 'Хрещатик',
             'audioDescription': str(tmp_path / 'a.wav')},
            {'textDescription': 'Сектор А',
             'audioDescription': str(tmp_path / 'b.wav')},
        ],
    }


def test_missing_clips_are_synthesized_once(monkeypatch, tmp_path):
    calls = []

    def fake_synthesize(jobs, **kwargs):
        jobs = list(jobs)
        calls.append(jobs)
        for job in jobs:
            job.output_path.write_bytes(b'RIFF')
        return jobs, []

    monkeypatch.setattr(engine, 'synthesize', fake_synthesize)
    _player(_model(tmp_path))._synthesize_missing()

    assert len(calls) == 1, 'the engine must be called once, not once per clip'
    assert sorted(j.output_path.name for j in calls[0]) == ['a.wav', 'b.wav']
    assert sorted(j.text for j in calls[0]) == ['Сектор А', 'Хрещатик']


def test_clips_that_exist_are_left_alone(monkeypatch, tmp_path):
    (tmp_path / 'a.wav').write_bytes(b'RIFF')
    calls = []

    def fake_synthesize(jobs, **kwargs):
        jobs = list(jobs)
        calls.append(jobs)
        return jobs, []

    monkeypatch.setattr(engine, 'synthesize', fake_synthesize)
    _player(_model(tmp_path))._synthesize_missing()
    assert [j.output_path.name for j in calls[0]] == ['b.wav']


def test_nothing_missing_means_no_engine_call(monkeypatch, tmp_path):
    (tmp_path / 'a.wav').write_bytes(b'RIFF')
    (tmp_path / 'b.wav').write_bytes(b'RIFF')

    def explode(jobs, **kwargs):
        raise AssertionError('the engine must not be called')

    monkeypatch.setattr(engine, 'synthesize', explode)
    _player(_model(tmp_path))._synthesize_missing()


def test_tts_unavailable_is_survivable(monkeypatch, tmp_path):
    def unavailable(jobs, **kwargs):
        raise engine.TTSUnavailable('no piper here')

    monkeypatch.setattr(engine, 'synthesize', unavailable)
    _player(_model(tmp_path))._synthesize_missing()   # must not raise


def test_disabling_the_fallback_skips_the_engine(monkeypatch, tmp_path):
    def explode(jobs, **kwargs):
        raise AssertionError('the engine must not be called')

    monkeypatch.setattr(engine, 'synthesize', explode)
    monkeypatch.setattr(audio_module.TTSConfig, 'RUNTIME_FALLBACK', False)
    _player(_model(tmp_path))._synthesize_missing()


def test_entries_without_text_are_not_synthesized(monkeypatch, tmp_path):
    calls = []

    def fake_synthesize(jobs, **kwargs):
        jobs = list(jobs)
        calls.append(jobs)
        return jobs, []

    monkeypatch.setattr(engine, 'synthesize', fake_synthesize)
    model = {'hotspots': [
        {'textDescription': '', 'audioDescription': str(tmp_path / 'a.wav')},
    ]}
    _player(model)._synthesize_missing()
    assert calls == []
