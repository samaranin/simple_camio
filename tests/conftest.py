"""Fixtures shared by the TTS tests."""

import stat
import textwrap

import pytest

STUB_PIPER = textwrap.dedent('''
    #!/usr/bin/env python3
    """Stand-in for the piper binary: writes a tiny valid WAV.

    Accepts the same flags the engine passes and reads its text from stdin. Text
    containing BOOM makes it fail, so a test can exercise one clip failing while
    its neighbours succeed.
    """
    import argparse
    import os
    import sys
    import wave

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output_file', required=True)
    args, _ = parser.parse_known_args()

    text = sys.stdin.read()
    if 'boom' in text.lower():
        sys.stderr.write('stub piper: refusing to synthesize\\n')
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    # Record the text verbatim so a test can assert what reached piper.
    with open(args.output_file + '.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    with wave.open(args.output_file, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b'\\x00\\x00' * 256)
''').lstrip()


@pytest.fixture
def stub_piper(tmp_path):
    """An executable that behaves enough like piper to test the engine."""
    path = tmp_path / 'stub_piper'
    path.write_text(STUB_PIPER, encoding='utf-8')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def stub_voice(tmp_path):
    """A voices directory holding the two files the engine checks for."""
    voices = tmp_path / 'voices'
    voices.mkdir()
    (voices / 'test_voice.onnx').write_bytes(b'not a real model')
    (voices / 'test_voice.onnx.json').write_text('{}', encoding='utf-8')
    return voices


@pytest.fixture
def sample_model_file(tmp_path):
    """
    A model JSON formatted the way the real ones are: compact hotspots, one per
    line, four-space outer indent, no trailing newline.
    """
    text = (
        '{\n'
        '    "model": {\n'
        '        "name": "Test",\n'
        '        "map_description": "audio/Description.mp3",\n'
        '        "mapDescriptionText": "Опис карти",\n'
        '        "hotspots":[\n'
        '        {"color":[255,0,0], "textDescription":"Хрещатик", '
        '"audioDescription":"audio/Khreschatyk.mp3"},\n'
        '        {"color":[0,255,0], "textDescription":"Сектор А", '
        '"audioDescription":"audio/Sector.mp3"}\n'
        '        ]\n'
        '    }\n'
        '}'
    )
    path = tmp_path / 'model.json'
    path.write_text(text, encoding='utf-8')
    return path
