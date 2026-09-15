"""
The model files are hand-formatted, and the generator rewrites paths in them on
every run. Reformatting would bury one changed value in a 231-line diff, so edits
are textual and verified rather than round-tripped through json.dump.
"""

import json

import pytest

from src.tts import json_edit

COMPACT = (
    '{\n'
    '    "model": {\n'
    '        "hotspots":[\n'
    '        {"color":[255,0,0], "textDescription":"Хрещатик", '
    '"audioDescription":"audio/Khreschatyk.mp3"},\n'
    '        {"color":[0,255,0], "textDescription":"Сектор А", '
    '"audioDescription":"audio/Sector.mp3"}\n'
    '        ]\n'
    '    }\n'
    '}'
)


def test_replacing_a_value_leaves_every_other_byte_alone():
    edited = json_edit.replace_string_value(
        COMPACT, 'audioDescription', 'audio/Khreschatyk.mp3', 'audio/tts/Khreschatyk.wav')
    assert 'audio/tts/Khreschatyk.wav' in edited
    # Only the one value changed: same line count, and the untouched hotspot is intact.
    assert len(edited.split('\n')) == len(COMPACT.split('\n'))
    assert '"audioDescription":"audio/Sector.mp3"' in edited


def test_cyrillic_is_written_literally_not_escaped():
    edited = json_edit.replace_string_value(
        COMPACT, 'textDescription', 'Хрещатик', 'Вулиця Хрещатик')
    assert 'Вулиця Хрещатик' in edited
    assert '\\u0425' not in edited


def test_a_value_that_appears_twice_is_refused():
    """
    Ambiguity means the SAME key and value twice - two hotspots pointing at one
    audio file, say. Two different keys sharing a value is not ambiguous, because
    the needle is the key/value pair, not the value alone.
    """
    text = '[{"a":"same"},{"a":"same"}]'
    with pytest.raises(json_edit.JsonEditError, match='2 times'):
        json_edit.replace_string_value(text, 'a', 'same', 'different')


def test_two_keys_sharing_a_value_is_not_ambiguous():
    text = '{"a":"same", "b":"same"}'
    assert json_edit.replace_string_value(text, 'a', 'same', 'different') == \
        '{"a":"different", "b":"same"}'


def test_a_value_that_is_not_there_is_refused():
    with pytest.raises(json_edit.JsonEditError, match='could not find'):
        json_edit.replace_string_value(COMPACT, 'audioDescription', 'nope.mp3', 'x.wav')


def test_both_spacings_after_the_colon_are_handled():
    spaced = '{"key": "value"}'
    assert json_edit.replace_string_value(spaced, 'key', 'value', 'other') == \
        '{"key": "other"}'


def test_write_verified_writes_when_the_data_matches(tmp_path):
    target = tmp_path / 'model.json'
    text = '{"a":"one"}'
    json_edit.write_verified(target, text, {'a': 'one'})
    assert target.read_text(encoding='utf-8') == text


def test_write_verified_refuses_when_the_data_does_not_match(tmp_path):
    target = tmp_path / 'model.json'
    with pytest.raises(json_edit.JsonEditError, match='expected data'):
        json_edit.write_verified(target, '{"a":"one"}', {'a': 'two'})
    assert not target.exists()


def test_write_verified_refuses_broken_json(tmp_path):
    target = tmp_path / 'model.json'
    with pytest.raises(json_edit.JsonEditError, match='not valid JSON'):
        json_edit.write_verified(target, '{"a":', {'a': 'one'})
    assert not target.exists()


def test_write_verified_adds_no_trailing_newline(tmp_path):
    """The real model files end without one; keep it that way."""
    target = tmp_path / 'model.json'
    json_edit.write_verified(target, '{"a":"one"}', {'a': 'one'})
    assert not target.read_text(encoding='utf-8').endswith('\n')


# Frozen from models/CnapMap/CnapFirstFloor.json as it read before Task 10 ran
# src.tts.generate_audio on the real models. This is the shape that broke round
# 1: the map's "map_description" and one hotspot's "audioDescription" genuinely
# reused one audio file. A synthetic fixture wouldn't have caught that
# regression, so it was originally read live off the real file; Task 10 now
# generates real narration for that same file and permanently repoints both
# keys at distinct generated clips, so the real file no longer has the shared
# value to read. Freezing the pre-generation shape here keeps the regression
# coverage without depending on mutable repository state.
REAL_CNAP_SNIPPET = (
    '{\n'
    '  "model":{\n'
    '    "map_description":"models/CnapMap/Audio/Passport Services.mp3",\n'
    '    "hotspots":[\n'
    '        {"color":[142,124,106], "colorComment":"color1", '
    '"textDescription": "Паспортні послуги",'
    '"audioDescription":"models/CnapMap/Audio/Passport Services.mp3"}\n'
    '    ]\n'
    '  }\n'
    '}'
)


def test_a_real_value_shared_across_two_keys_edits_only_the_intended_one():
    raw = REAL_CNAP_SNIPPET
    shared_value = 'models/CnapMap/Audio/Passport Services.mp3'
    assert raw.count(json.dumps(shared_value, ensure_ascii=False)) == 2, \
        'fixture assumption changed: the value is no longer shared by two keys'

    edited = json_edit.replace_string_value(
        raw, 'audioDescription', shared_value,
        'models/CnapMap/Audio/tts/Passport Services.wav')

    # The hotspot's audioDescription changed...
    assert '"audioDescription":"models/CnapMap/Audio/tts/Passport Services.wav"' in edited
    # ...but the map's map_description, which shared the same value, did not.
    assert f'"map_description":"{shared_value}"' in edited

    before, after = raw.split('\n'), edited.split('\n')
    changed = [i for i, (a, b) in enumerate(zip(before, after), 1) if a != b]
    assert len(before) == len(after) and len(changed) == 1
