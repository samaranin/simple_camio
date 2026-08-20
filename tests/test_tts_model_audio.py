"""Planning is pure, so it is tested with a dict and a tmp_path."""

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
