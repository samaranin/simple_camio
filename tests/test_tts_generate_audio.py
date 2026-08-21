"""End-to-end over the CLI's function, with the stub piper standing in."""

import json
import shutil
from pathlib import Path

import pytest

from src.tts import generate_audio


def _run(model_file, stub_piper, stub_voice, **kwargs):
    return generate_audio.generate_for_model(
        model_file, piper_bin=str(stub_piper), voice='test_voice',
        voices_dir=stub_voice, **kwargs)


def test_generates_every_missing_clip(sample_model_file, stub_piper, stub_voice,
                                       monkeypatch):
    # The model stores its audio paths relative to the directory it lives in (as
    # the real models do, relative to the repo root they are run from), so the
    # process cwd has to match that directory for a relative path to resolve.
    monkeypatch.chdir(sample_model_file.parent)
    failed = _run(sample_model_file, stub_piper, stub_voice)
    assert failed == 0
    audio = sample_model_file.parent / 'audio' / 'tts'
    # stub_piper also writes a `.txt` companion beside each `.wav` (see conftest.py),
    # so only the generated clips themselves are asserted here.
    assert sorted(p.name for p in audio.glob('*.wav')) == \
        ['Khreschatyk.wav', 'Sector.wav', 'map_description.wav']


def test_rewrites_only_the_paths_it_generated(sample_model_file, stub_piper, stub_voice,
                                               monkeypatch):
    monkeypatch.chdir(sample_model_file.parent)
    _run(sample_model_file, stub_piper, stub_voice)
    model = json.loads(sample_model_file.read_text(encoding='utf-8'))['model']
    assert model['hotspots'][0]['audioDescription'] == 'audio/tts/Khreschatyk.wav'
    assert model['hotspots'][1]['audioDescription'] == 'audio/tts/Sector.wav'
    assert model['map_description'] == 'audio/tts/map_description.wav'


def test_keeps_the_files_formatting(sample_model_file, stub_piper, stub_voice,
                                     monkeypatch):
    monkeypatch.chdir(sample_model_file.parent)
    before = sample_model_file.read_text(encoding='utf-8')
    _run(sample_model_file, stub_piper, stub_voice)
    after = sample_model_file.read_text(encoding='utf-8')
    assert len(before.split('\n')) == len(after.split('\n'))
    assert not after.endswith('\n')
    assert '\\u0425' not in after


def test_a_second_run_generates_nothing(sample_model_file, stub_piper, stub_voice,
                                         monkeypatch):
    monkeypatch.chdir(sample_model_file.parent)
    _run(sample_model_file, stub_piper, stub_voice)
    after_first = sample_model_file.read_text(encoding='utf-8')
    stamps = {p: p.stat().st_mtime_ns
              for p in (sample_model_file.parent / 'audio' / 'tts').iterdir()}

    assert _run(sample_model_file, stub_piper, stub_voice) == 0
    assert sample_model_file.read_text(encoding='utf-8') == after_first
    assert {p: p.stat().st_mtime_ns for p in stamps} == stamps


def test_force_regenerates(sample_model_file, stub_piper, stub_voice, monkeypatch):
    monkeypatch.chdir(sample_model_file.parent)
    _run(sample_model_file, stub_piper, stub_voice)
    target = sample_model_file.parent / 'audio' / 'tts' / 'Khreschatyk.wav'
    target.write_bytes(b'clobbered')
    _run(sample_model_file, stub_piper, stub_voice, force=True)
    assert target.read_bytes() != b'clobbered'


def test_a_failing_clip_is_counted_and_its_path_is_not_rewritten(
        tmp_path, stub_piper, stub_voice, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = (
        '{\n'
        '    "model": {\n'
        '        "hotspots":[\n'
        '        {"textDescription":"BOOM", "audioDescription":"audio/bad.mp3"},\n'
        '        {"textDescription":"Хрещатик", "audioDescription":"audio/good.mp3"}\n'
        '        ]\n'
        '    }\n'
        '}'
    )
    model_file = tmp_path / 'model.json'
    model_file.write_text(text, encoding='utf-8')

    failed = _run(model_file, stub_piper, stub_voice)
    assert failed == 1
    model = json.loads(model_file.read_text(encoding='utf-8'))['model']
    assert model['hotspots'][0]['audioDescription'] == 'audio/bad.mp3'
    assert model['hotspots'][1]['audioDescription'] == 'audio/tts/good.wav'


def test_an_entry_with_no_text_is_skipped_not_synthesized(
        tmp_path, stub_piper, stub_voice, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = (
        '{\n'
        '    "model": {\n'
        '        "hotspots":[\n'
        '        {"textDescription":"", "audioDescription":"audio/empty.mp3"}\n'
        '        ]\n'
        '    }\n'
        '}'
    )
    model_file = tmp_path / 'model.json'
    model_file.write_text(text, encoding='utf-8')
    assert _run(model_file, stub_piper, stub_voice) == 0
    assert not (tmp_path / 'audio').exists()


def test_main_returns_nonzero_when_something_failed(tmp_path, stub_piper, stub_voice,
                                                    monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = (
        '{\n'
        '    "model": {\n'
        '        "hotspots":[\n'
        '        {"textDescription":"BOOM", "audioDescription":"audio/bad.mp3"}\n'
        '        ]\n'
        '    }\n'
        '}'
    )
    model_file = tmp_path / 'model.json'
    model_file.write_text(text, encoding='utf-8')
    code = generate_audio.main([
        '--input1', str(model_file),
        '--piper-bin', str(stub_piper),
        '--voice', 'test_voice',
        '--voices-dir', str(stub_voice),
    ])
    assert code == 1


def test_the_real_cnap_model_gets_distinct_audio_for_its_shared_source_file(
        tmp_path, stub_piper, stub_voice, monkeypatch):
    """
    CnapFirstFloor.json points map_description at the same source file as its
    first hotspot ("Passport Services.mp3"). Without Step 0's fixed output name,
    both texts would collide on one output file. The real file already carries
    its own mapDescriptionText (Task 9), but it is overwritten with a fixed
    value here so the test does not depend on that text's content.

    The real file is copied into tmp_path before anything writes, and nothing
    under the repo's models/ directory is ever touched. Task 10 has since run
    the real generator over models/CnapMap/CnapFirstFloor.json, so
    models/CnapMap/Audio/tts/ now holds real generated audio; this test
    snapshots that directory and the real model file instead of asserting they
    stay absent/unmodified from a clean checkout.
    """
    # Resolved to absolute before any chdir, so these always name the real
    # repository's paths regardless of what the process's cwd becomes below.
    real_model_path = Path.cwd() / 'models' / 'CnapMap' / 'CnapFirstFloor.json'
    real_audio_tts_dir = Path.cwd() / 'models' / 'CnapMap' / 'Audio' / 'tts'
    before_real_model = real_model_path.read_text(encoding='utf-8')
    before_stamps = {
        p: p.stat().st_mtime_ns for p in real_audio_tts_dir.iterdir()
    } if real_audio_tts_dir.exists() else {}

    copied_model_path = tmp_path / 'models' / 'CnapMap' / 'CnapFirstFloor.json'
    copied_model_path.parent.mkdir(parents=True)
    shutil.copy(real_model_path, copied_model_path)

    document = json.loads(copied_model_path.read_text(encoding='utf-8'))
    document['model']['mapDescriptionText'] = 'опис карти'
    copied_model_path.write_text(json.dumps(document, ensure_ascii=False),
                                  encoding='utf-8')

    monkeypatch.chdir(tmp_path)
    failed = _run(copied_model_path, stub_piper, stub_voice)
    assert failed == 0

    model = json.loads(copied_model_path.read_text(encoding='utf-8'))['model']
    hotspot_audio = model['hotspots'][0]['audioDescription']
    description_audio = model['map_description']

    assert model['hotspots'][0]['textDescription'] == 'Паспортні послуги'
    assert hotspot_audio != description_audio
    assert (tmp_path / hotspot_audio).is_file()
    assert (tmp_path / description_audio).is_file()

    # The real repository is untouched: everything landed under tmp_path.
    assert real_model_path.read_text(encoding='utf-8') == before_real_model
    after_stamps = {
        p: p.stat().st_mtime_ns for p in real_audio_tts_dir.iterdir()
    } if real_audio_tts_dir.exists() else {}
    assert after_stamps == before_stamps
