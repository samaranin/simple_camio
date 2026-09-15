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
    """
    audioDescription already points inside the generated subdirectory
    (model_audio.TTSConfig.GENERATED_SUBDIR, 'tts'), matching every real model
    that has already been through the CLI once - so model_audio.output_path()
    resolves to the same path named here, and these tests exercise the same
    "nothing changes for an already-generated model" case the real fallback
    hits in the field.
    """
    return {
        'hotspots': [
            {'textDescription': 'Хрещатик',
             'audioDescription': str(tmp_path / 'tts' / 'a.wav')},
            {'textDescription': 'Сектор А',
             'audioDescription': str(tmp_path / 'tts' / 'b.wav')},
        ],
    }


#: Content long enough to clear model_audio.MIN_VALID_WAV_BYTES, standing in
#: for a real WAV so "already exists" tests do not trip the interrupted-run
#: self-heal check instead.
_REAL_ENOUGH_CONTENT = b'a real clip, well past the minimum valid WAV size'


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
    (tmp_path / 'tts').mkdir()
    (tmp_path / 'tts' / 'a.wav').write_bytes(_REAL_ENOUGH_CONTENT)
    calls = []

    def fake_synthesize(jobs, **kwargs):
        jobs = list(jobs)
        calls.append(jobs)
        return jobs, []

    monkeypatch.setattr(engine, 'synthesize', fake_synthesize)
    _player(_model(tmp_path))._synthesize_missing()
    assert [j.output_path.name for j in calls[0]] == ['b.wav']


def test_nothing_missing_means_no_engine_call(monkeypatch, tmp_path):
    (tmp_path / 'tts').mkdir()
    (tmp_path / 'tts' / 'a.wav').write_bytes(_REAL_ENOUGH_CONTENT)
    (tmp_path / 'tts' / 'b.wav').write_bytes(_REAL_ENOUGH_CONTENT)

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


def test_the_destination_comes_from_model_audio_not_the_literal_json_path(
        monkeypatch, tmp_path):
    """
    A model that has never been through the CLI still names an un-generated
    path, e.g. Audio/Khreschatyk.mp3. The fallback must write the generated
    clip under model_audio's generated subdirectory (Audio/tts/Khreschatyk.wav)
    rather than as WAV bytes under the literal .mp3 name it was told - the
    same destination the offline generator CLI would use, so one module
    (model_audio) owns where a generated clip belongs regardless of which
    path produced it.
    """
    calls = []

    def fake_synthesize(jobs, **kwargs):
        jobs = list(jobs)
        calls.append(jobs)
        for job in jobs:
            job.output_path.parent.mkdir(parents=True, exist_ok=True)
            job.output_path.write_bytes(_REAL_ENOUGH_CONTENT)
        return jobs, []

    monkeypatch.setattr(engine, 'synthesize', fake_synthesize)
    model = {'hotspots': [
        {'textDescription': 'Хрещатик',
         'audioDescription': str(tmp_path / 'Audio' / 'Khreschatyk.mp3')},
    ]}
    _player(model)._synthesize_missing()

    assert calls[0][0].output_path == tmp_path / 'Audio' / 'tts' / 'Khreschatyk.wav'


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
