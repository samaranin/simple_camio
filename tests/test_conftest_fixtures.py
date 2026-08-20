"""The stub piper has to actually behave like piper, or every test below it lies."""

import json
import subprocess
import wave


def test_stub_piper_writes_a_valid_wav(stub_piper, tmp_path):
    out = tmp_path / 'out.wav'
    result = subprocess.run(
        [str(stub_piper), '--model', 'ignored', '--output_file', str(out)],
        input='Хрещатик', text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    with wave.open(str(out)) as w:
        assert w.getnframes() == 256


def test_stub_piper_fails_on_boom(stub_piper, tmp_path):
    out = tmp_path / 'out.wav'
    result = subprocess.run(
        [str(stub_piper), '--model', 'ignored', '--output_file', str(out)],
        input='BOOM', text=True, capture_output=True,
    )
    assert result.returncode == 1
    assert not out.exists()


def test_sample_model_parses_and_keeps_its_formatting(sample_model_file):
    raw = sample_model_file.read_text(encoding='utf-8')
    assert not raw.endswith('\n')
    model = json.loads(raw)['model']
    assert len(model['hotspots']) == 2
    assert model['hotspots'][0]['textDescription'] == 'Хрещатик'
