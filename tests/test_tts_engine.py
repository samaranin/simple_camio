"""The engine is the only thing that talks to piper, so it owns the failure modes."""

from pathlib import Path

import pytest

from src.tts import engine


def test_resolve_piper_rejects_a_path_that_is_not_there():
    with pytest.raises(engine.TTSUnavailable, match='not found'):
        engine.resolve_piper('/nonexistent/piper')


def test_resolve_piper_finds_it_beside_the_interpreter(monkeypatch, tmp_path):
    """
    piper installs into .venv/bin/, which is not on PATH because the project runs
    as .venv/bin/python without activating the venv. shutil.which alone misses it.
    """
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    piper = fake_bin / 'piper'
    piper.write_text('#!/bin/sh\n')
    piper.chmod(0o755)

    monkeypatch.setattr(engine.sys, 'executable', str(fake_bin / 'python'))
    monkeypatch.setattr(engine.shutil, 'which', lambda name: None)
    assert engine.resolve_piper() == str(piper)


def test_resolve_voice_names_the_files_it_wanted(tmp_path):
    with pytest.raises(engine.TTSUnavailable, match='uk_UA'):
        engine.resolve_voice(voice='uk_UA-missing', voices_dir=tmp_path)


def test_resolve_voice_returns_both_files(stub_voice):
    onnx, config = engine.resolve_voice(voice='test_voice', voices_dir=stub_voice)
    assert onnx.name == 'test_voice.onnx'
    assert config.name == 'test_voice.onnx.json'


def test_no_jobs_is_not_an_error():
    assert engine.synthesize([]) == ([], [])


def test_synthesize_writes_one_wav_per_job(stub_piper, stub_voice, tmp_path):
    jobs = [
        engine.SynthesisJob(text='Хрещатик', output_path=tmp_path / 'out' / 'a.wav'),
        engine.SynthesisJob(text='Сектор А', output_path=tmp_path / 'out' / 'b.wav'),
    ]
    ok, failed = engine.synthesize(
        jobs, piper_bin=str(stub_piper), voice='test_voice', voices_dir=stub_voice)
    assert failed == []
    assert [j.output_path.name for j in ok] == ['a.wav', 'b.wav']
    assert all(j.output_path.is_file() for j in jobs)


def test_one_failing_clip_does_not_stop_the_others(stub_piper, stub_voice, tmp_path):
    jobs = [
        engine.SynthesisJob(text='Хрещатик', output_path=tmp_path / 'a.wav'),
        engine.SynthesisJob(text='BOOM', output_path=tmp_path / 'b.wav'),
        engine.SynthesisJob(text='Сектор А', output_path=tmp_path / 'c.wav'),
    ]
    ok, failed = engine.synthesize(
        jobs, piper_bin=str(stub_piper), voice='test_voice', voices_dir=stub_voice)
    assert [j.output_path.name for j in ok] == ['a.wav', 'c.wav']
    assert [j.output_path.name for j in failed] == ['b.wav']


def test_capital_cyrillic_is_lowercased_before_it_reaches_piper(
        stub_piper, stub_voice, tmp_path):
    """
    The uk_UA voices' phoneme maps lack capital Cyrillic, so 'Хрещатик' loses its
    leading sound. Every zone name is capitalised, so this must not reach piper.
    """
    out = tmp_path / 'a.wav'
    engine.synthesize(
        [engine.SynthesisJob(text='Хрещатик', output_path=out)],
        piper_bin=str(stub_piper), voice='test_voice', voices_dir=stub_voice)
    assert out.with_suffix('.wav.txt').read_text(encoding='utf-8') == 'хрещатик'


def test_progress_is_reported_per_clip(stub_piper, stub_voice, tmp_path):
    seen = []
    jobs = [engine.SynthesisJob(text=f'текст {i}', output_path=tmp_path / f'{i}.wav')
            for i in range(3)]
    engine.synthesize(jobs, piper_bin=str(stub_piper), voice='test_voice',
                      voices_dir=stub_voice,
                      progress=lambda done, total, job: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_missing_piper_raises_before_any_work(stub_voice, tmp_path):
    jobs = [engine.SynthesisJob(text='Хрещатик', output_path=tmp_path / 'a.wav')]
    with pytest.raises(engine.TTSUnavailable):
        engine.synthesize(jobs, piper_bin='/nonexistent/piper',
                          voice='test_voice', voices_dir=stub_voice)
    assert not (tmp_path / 'a.wav').exists()
