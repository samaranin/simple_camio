"""
Every zone must have Ukrainian text, or TTS has nothing to say for it. This is a
guard against a hotspot being added later with a transliterated label.
"""

import glob
import json
import unicodedata

import pytest


def _model_files():
    """
    Map models under models/, identified by having a "model" section.

    A bare glob also matches the Piper voice config that Task 1 downloads to
    models/tts_voices/, which has no hotspots and would fail every test here.
    """
    found = []
    for path in sorted(glob.glob('models/*/*.json')):
        with open(path, encoding='utf-8') as f:
            if 'model' in json.load(f):
                found.append(path)
    return found


MODEL_FILES = _model_files()


def _is_cyrillic(text):
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(
        'CYRILLIC' in unicodedata.name(c, '') for c in letters)


@pytest.mark.parametrize('path', MODEL_FILES)
def test_every_hotspot_has_cyrillic_text(path):
    model = json.loads(open(path, encoding='utf-8').read())['model']
    offenders = [
        (i, h.get('textDescription', ''))
        for i, h in enumerate(model.get('hotspots', []))
        if not _is_cyrillic(h.get('textDescription', ''))
    ]
    assert offenders == [], f'{path}: not Cyrillic: {offenders}'


@pytest.mark.parametrize('path', MODEL_FILES)
def test_every_model_has_map_description_text(path):
    model = json.loads(open(path, encoding='utf-8').read())['model']
    if not model.get('map_description'):
        pytest.skip('this model has no map description')
    assert _is_cyrillic(model.get('mapDescriptionText', ''))
