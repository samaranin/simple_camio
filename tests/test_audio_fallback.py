"""
The fallback runs once at load, so a finger resting on a zone never waits for
synthesis, and a broken TTS setup never takes the player down with it.
"""

from pathlib import Path

import pytest

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


def test_existing_recording_outside_tts_is_reused(monkeypatch, tmp_path):
    recording = tmp_path / 'Recordings' / 'zone-1.wav'
    recording.parent.mkdir()
    recording.write_bytes(_REAL_ENOUGH_CONTENT)

    def explode(jobs, **kwargs):
        raise AssertionError('a ready recording must not trigger TTS')

    monkeypatch.setattr(engine, 'synthesize', explode)
    model = {'hotspots': [{'textDescription': 'Зона', 'audioDescription': str(recording)}]}
    _player(model)._synthesize_missing()
    assert not (recording.parent / 'tts').exists()


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


def test_generated_paths_are_loaded_after_partial_synthesis(monkeypatch, tmp_path):
    model = {'hotspots': [
        {'color': [255, 0, 0], 'textDescription': 'Зона',
         'audioDescription': str(tmp_path / 'Audio/zone.mp3')},
        {'color': [0, 255, 0], 'textDescription': 'Інша зона',
         'audioDescription': str(tmp_path / 'Audio/failed.mp3')},
    ], 'map_description': str(tmp_path / 'Audio/description.mp3'),
       'mapDescriptionText': 'Опис карти'}
    original_failed = model['hotspots'][1]['audioDescription']

    def synthesize(jobs, **kwargs):
        jobs = list(jobs)
        for job in (jobs[0], jobs[2]):
            job.output_path.parent.mkdir(parents=True, exist_ok=True)
            job.output_path.write_bytes(_REAL_ENOUGH_CONTENT)
        return [jobs[0], jobs[2]], [jobs[1]]

    monkeypatch.setattr(engine, 'synthesize', synthesize)
    player = _player(model)
    player._synthesize_missing()
    assert model['hotspots'][0]['audioDescription'] == str(tmp_path / 'Audio/tts/zone.wav')
    assert model['hotspots'][1]['audioDescription'] == original_failed
    assert model['map_description'] == str(tmp_path / 'Audio/tts/map_description.wav')
    loaded = []
    player.hotspots = {}
    player.sound_files = {}
    player._load_sound = lambda path, label: loaded.append(Path(path))
    player._load_hotspot_audio()
    assert loaded[0].is_file()
    assert loaded[1] == Path(original_failed)


@pytest.mark.parametrize('fallback', [True, False])
def test_existing_generated_clip_is_reused_without_piper(monkeypatch, tmp_path, fallback):
    generated = tmp_path / 'Audio/tts/zone.wav'
    generated.parent.mkdir(parents=True)
    generated.write_bytes(_REAL_ENOUGH_CONTENT)
    model = {'hotspots': [{'textDescription': 'Зона',
                          'audioDescription': str(tmp_path / 'Audio/zone.mp3')}]}
    monkeypatch.setattr(audio_module.TTSConfig, 'RUNTIME_FALLBACK', fallback)

    def explode(*args, **kwargs):
        raise AssertionError('an existing generated clip must not invoke Piper')

    monkeypatch.setattr(engine, 'synthesize', explode)
    _player(model)._synthesize_missing()
    assert model['hotspots'][0]['audioDescription'] == str(generated)


def test_unavailable_piper_does_not_replace_missing_path(monkeypatch, tmp_path):
    model = {'hotspots': [{'textDescription': 'Зона',
                          'audioDescription': str(tmp_path / 'Audio/zone.mp3')}]}
    original = model['hotspots'][0]['audioDescription']

    def unavailable(*args, **kwargs):
        raise engine.TTSUnavailable('not installed')

    monkeypatch.setattr(engine, 'synthesize', unavailable)
    _player(model)._synthesize_missing()
    assert model['hotspots'][0]['audioDescription'] == original


@pytest.mark.parametrize('fallback', [True, False])
@pytest.mark.parametrize('existing_generated', [True, False])
@pytest.mark.parametrize('overview_collision', [True, False])
def test_colliding_narrations_are_neither_generated_nor_adopted(
        monkeypatch, tmp_path, fallback, existing_generated, overview_collision):
    stem = 'map_description' if overview_collision else 'same'
    first = str(tmp_path / f'Audio/{stem}.mp3')
    second = str(tmp_path / f'Audio/{stem}.ogg')
    model = {'hotspots': [{'textDescription': 'Перша зона', 'audioDescription': first}]}
    if overview_collision:
        model.update(map_description=second, mapDescriptionText='Опис карти')
    else:
        # An entry without text must not silently adopt the other zone's clip.
        model['hotspots'].append({'textDescription': '', 'audioDescription': second})
    generated = tmp_path / f'Audio/tts/{stem}.wav'
    if existing_generated:
        generated.parent.mkdir(parents=True)
        generated.write_bytes(_REAL_ENOUGH_CONTENT)
    monkeypatch.setattr(audio_module.TTSConfig, 'RUNTIME_FALLBACK', fallback)

    def explode(*args, **kwargs):
        raise AssertionError('colliding narrations must not invoke Piper')

    monkeypatch.setattr(engine, 'synthesize', explode)
    _player(model)._synthesize_missing()
    assert model['hotspots'][0]['audioDescription'] == first
    assert (model['map_description'] if overview_collision
            else model['hotspots'][1]['audioDescription']) == second
    assert generated.exists() == existing_generated
    if existing_generated:
        assert generated.read_bytes() == _REAL_ENOUGH_CONTENT


def test_collisions_do_not_block_independent_narration(monkeypatch, tmp_path):
    model = {'hotspots': [
        {'textDescription': 'Перша зона', 'audioDescription': str(tmp_path / 'same.mp3')},
        {'textDescription': 'Друга зона', 'audioDescription': str(tmp_path / 'same.ogg')},
        {'textDescription': 'Третя зона', 'audioDescription': str(tmp_path / 'unique.mp3')},
    ]}
    calls = []

    def synthesize(jobs, **kwargs):
        jobs = list(jobs)
        calls.extend(jobs)
        for job in jobs:
            job.output_path.parent.mkdir(parents=True, exist_ok=True)
            job.output_path.write_bytes(_REAL_ENOUGH_CONTENT)
        return jobs, []

    monkeypatch.setattr(engine, 'synthesize', synthesize)
    _player(model)._synthesize_missing()
    assert [job.output_path.name for job in calls] == ['unique.wav']
    assert model['hotspots'][2]['audioDescription'] == str(tmp_path / 'tts/unique.wav')
    assert not (tmp_path / 'tts/same.wav').exists()


def test_distinct_valid_recordings_with_same_stem_are_preserved(monkeypatch, tmp_path):
    paths = [tmp_path / 'same.mp3', tmp_path / 'same.ogg']
    for path in paths:
        path.write_bytes(_REAL_ENOUGH_CONTENT)
    model = {'hotspots': [{'textDescription': str(index), 'audioDescription': str(path)}
                          for index, path in enumerate(paths)]}

    def explode(*args, **kwargs):
        raise AssertionError('valid original recordings must not invoke Piper')

    monkeypatch.setattr(engine, 'synthesize', explode)
    _player(model)._synthesize_missing()
    assert [hotspot['audioDescription'] for hotspot in model['hotspots']] == [str(path) for path in paths]
    assert all(path.read_bytes() == _REAL_ENOUGH_CONTENT for path in paths)


def test_generated_replacement_cannot_adopt_another_zones_recording(monkeypatch, tmp_path):
    recording = tmp_path / 'tts/shared.wav'
    recording.parent.mkdir()
    recording.write_bytes(_REAL_ENOUGH_CONTENT)
    missing = str(tmp_path / 'shared.mp3')
    model = {'hotspots': [
        {'textDescription': 'Перша', 'audioDescription': missing},
        {'textDescription': 'Друга', 'audioDescription': str(recording)},
    ]}

    def explode(*args, **kwargs):
        raise AssertionError('owned recording cannot be replaced')

    monkeypatch.setattr(engine, 'synthesize', explode)
    _player(model)._synthesize_missing()
    assert model['hotspots'][0]['audioDescription'] == missing
    assert model['hotspots'][1]['audioDescription'] == str(recording)
    assert recording.read_bytes() == _REAL_ENOUGH_CONTENT
