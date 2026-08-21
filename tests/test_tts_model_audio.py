"""Planning is pure, so it is tested with a dict and a tmp_path."""

import json
from pathlib import Path

from src.tts import model_audio


def test_generated_path_keeps_the_existing_audio_directory():
    assert model_audio.generated_path('models/UkraineMap/Audio/Khreschyatik.mp3') == \
        Path('models/UkraineMap/Audio/tts/Khreschyatik.wav')


def test_generated_path_handles_the_heart_models_sound_directory():
    """Heart calls its audio directory Sound/, not Audio/."""
    assert model_audio.generated_path('models/Heart/Sound/Aorta.mp3') == \
        Path('models/Heart/Sound/tts/Aorta.wav')


def test_narration_entries_lists_hotspots_in_order():
    model = {'hotspots': [
        {'textDescription': 'Хрещатик', 'audioDescription': 'a/one.mp3'},
        {'textDescription': 'Сектор А', 'audioDescription': 'a/two.mp3'},
    ]}
    entries = model_audio.narration_entries(model)
    assert [e.index for e in entries] == [0, 1]
    assert [e.text for e in entries] == ['Хрещатик', 'Сектор А']
    assert all(e.kind == 'hotspot' for e in entries)


def test_narration_entries_includes_the_map_description():
    model = {
        'hotspots': [],
        'map_description': 'a/desc.mp3',
        'mapDescriptionText': 'Опис карти',
    }
    entries = model_audio.narration_entries(model)
    assert len(entries) == 1
    assert entries[0].kind == 'map_description'
    assert entries[0].index == model_audio.MAP_DESCRIPTION_INDEX
    assert entries[0].text == 'Опис карти'


def test_a_map_description_with_no_text_is_still_listed_but_has_no_text():
    """It is listed so the CLI can warn about it rather than silently skip it."""
    model = {'hotspots': [], 'map_description': 'a/desc.mp3'}
    entries = model_audio.narration_entries(model)
    assert entries[0].text == ''


def test_pending_skips_entries_with_no_text():
    entries = [model_audio.NarrationEntry('hotspot', 0, '', 'a/one.mp3')]
    assert model_audio.pending(entries) == []


def test_pending_skips_output_that_already_exists(tmp_path):
    target = tmp_path / 'audio' / 'tts' / 'one.wav'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'already here')
    entries = [model_audio.NarrationEntry(
        'hotspot', 0, 'Хрещатик', str(tmp_path / 'audio' / 'one.mp3'))]
    assert model_audio.pending(entries) == []


def test_force_includes_output_that_already_exists(tmp_path):
    target = tmp_path / 'audio' / 'tts' / 'one.wav'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'already here')
    entries = [model_audio.NarrationEntry(
        'hotspot', 0, 'Хрещатик', str(tmp_path / 'audio' / 'one.mp3'))]
    assert len(model_audio.pending(entries, force=True)) == 1


def test_pending_includes_missing_output(tmp_path):
    entries = [model_audio.NarrationEntry(
        'hotspot', 0, 'Хрещатик', str(tmp_path / 'audio' / 'one.mp3'))]
    assert len(model_audio.pending(entries)) == 1


def test_output_path_keeps_a_hotspots_basename():
    entry = model_audio.NarrationEntry('hotspot', 0, 'Хрещатик', 'a/Khreschatyk.mp3')
    assert model_audio.output_path(entry) == Path('a/tts/Khreschatyk.wav')


def test_the_map_description_gets_a_fixed_name_not_the_shared_basename():
    """
    CnapMap and Heart point map_description at the same file as one of their
    hotspots, so deriving from the basename would collide.
    """
    entry = model_audio.NarrationEntry(
        'map_description', model_audio.MAP_DESCRIPTION_INDEX, 'Опис', 'a/Aorta.mp3')
    assert model_audio.output_path(entry) == Path('a/tts/map_description.wav')


def test_no_real_model_produces_two_entries_with_one_output_path():
    """The collision this function exists to prevent, checked on real data."""
    import glob
    for path in sorted(glob.glob('models/*/*.json')):
        with open(path, encoding='utf-8') as f:
            document = json.load(f)
        if 'model' not in document:
            continue
        model = document['model']
        model.setdefault('mapDescriptionText', 'опис карти')
        outs = [str(model_audio.output_path(e))
                for e in model_audio.narration_entries(model) if e.text]
        assert len(outs) == len(set(outs)), f'{path}: duplicate output paths'
