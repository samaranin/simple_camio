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
    text = '{"a":"same", "b":"same"}'
    with pytest.raises(json_edit.JsonEditError, match='2 times'):
        json_edit.replace_string_value(text, 'a', 'same', 'different')


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
