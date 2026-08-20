# TTS Zone Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate Ukrainian zone narration with Piper instead of recording an MP3 per hotspot, pre-generated off-device with live synthesis on the Raspberry Pi 4 as a fallback.

**Architecture:** A new `src/tts/` package owns everything Piper-related. A CLI walks a model JSON and writes WAV files ahead of time; `ZoneAudioPlayer` calls the same engine at load time to fill anything missing. Piper runs as a subprocess so its ONNX runtime never shares a process with MediaPipe.

**Tech Stack:** Python 3.12, piper-tts 1.7.0 (GPL-3.0), the `uk_UA-ukrainian_tts-medium` voice, pytest (new to this project), pyglet/pygame for playback.

**Spec:** `docs/superpowers/specs/2026-08-20-tts-zone-audio-design.md`

**One refinement against the spec:** the spec names four components; this plan has
five. Writing it surfaced that the spec's own test requirement — "the rest of the
file byte-identical" — cannot be met by `json.dump`, because the models are
hand-formatted with one hotspot per line and reformatting `UkraineMap.json` yields a
231-line diff. The formatting-preserving editing is therefore its own module,
`src/tts/json_edit.py` (Task 5), rather than being buried in the CLI.

## Global Constraints

- Every command runs from the repository root; modules import absolutely (`from src.config import ...`) and submodules run via `python -m`, never by file path.
- The interpreter is `.venv/bin/python`. Install with `uv pip install --python .venv/bin/python ...`; the venv has no `pip`.
- No TTS failure may crash the application. A missing clip leaves one zone silent; the main loop still starts.
- Existing audio is never overwritten without an explicit `--force`.
- Model JSON files are hand-formatted with one hotspot per line, 4-space outer indent, and **no trailing newline**. Never round-trip them through `json.dump`: on `UkraineMap.json` that turns 34 lines into 196 and yields a 231-line diff. Edit values in the raw text and verify by re-parsing.
- Model JSON is written with `ensure_ascii=False`. With escaping on, Cyrillic becomes `Х...` and the migration diff is unreadable.
- The voice model (`uk_UA-ukrainian_tts-medium.onnx`, 76.7 MB) is never committed.
- The runtime fallback writes audio files only. It never edits a model JSON — a device in the field does not rewrite its own configuration.
- `models/Heart/` keeps its audio in `Sound/`, the other two models use `Audio/`. Generated files go to a `tts/` subdirectory of whichever directory the existing path names.

## File Structure

**Create:**
- `src/tts/__init__.py` — package marker
- `src/tts/engine.py` — the only module that knows Piper exists: binary and voice resolution, subprocess invocation
- `src/tts/model_audio.py` — planning over a parsed model dict: which entries need audio, where output goes. Pure
- `src/tts/json_edit.py` — formatting-preserving value replacement in raw JSON text, with verification
- `src/tts/generate_audio.py` — CLI orchestration and reporting
- `requirements-dev.txt` — pytest
- `tests/conftest.py` — stub piper binary and sample-model fixtures
- `tests/test_tts_engine.py`, `tests/test_tts_model_audio.py`, `tests/test_tts_json_edit.py`, `tests/test_tts_generate_audio.py`, `tests/test_audio_loading.py`
- `docs/tts-setup.md` — verified install commands and measured timings

**Modify:**
- `src/config.py` — add `TTSConfig`
- `src/audio/audio.py` — extract `_load_sound()`, add `_synthesize_missing()`
- `.gitignore` — voice model directory
- `models/*/*.json` — Cyrillic `textDescription`, new `mapDescriptionText`, regenerated `audioDescription`
- `README.md`, `ARCHITECTURE.md` — the generator command and voice download

---

### Task 1: Verify the Piper toolchain and record the facts

The engine's shape depends on how Piper actually behaves. Establish that first so no later task is built on a guess.

**Files:**
- Create: `docs/tts-setup.md`

**Interfaces:**
- Consumes: nothing
- Produces: the verified voice name for `TTSConfig.VOICE` in Task 2, and the measured per-clip synthesis cost that Task 11 checks against

- [ ] **Step 1: Install piper into the venv**

```bash
uv pip install --python .venv/bin/python 'piper-tts==1.7.0'
.venv/bin/python -c "import piper; print('piper module OK')"
which piper || ls .venv/bin/piper
```

- [ ] **Step 2: Download the Ukrainian voice**

```bash
mkdir -p models/tts_voices
BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/ukrainian_tts/medium
curl -L -o models/tts_voices/uk_UA-ukrainian_tts-medium.onnx      "$BASE/uk_UA-ukrainian_tts-medium.onnx"
curl -L -o models/tts_voices/uk_UA-ukrainian_tts-medium.onnx.json "$BASE/uk_UA-ukrainian_tts-medium.onnx.json"
ls -la models/tts_voices/
```

Expected: a 76.7 MB `.onnx` and a small `.onnx.json`.

- [ ] **Step 3: Synthesize one Ukrainian phrase and confirm the WAV is real**

```bash
echo 'Хрещатик' | .venv/bin/piper \
  --model models/tts_voices/uk_UA-ukrainian_tts-medium.onnx \
  --output_file /tmp/probe.wav
.venv/bin/python -c "
import wave
with wave.open('/tmp/probe.wav') as w:
    print('channels', w.getnchannels(), 'rate', w.getframerate(), 'frames', w.getnframes())
    assert w.getnframes() > 1000, 'suspiciously short output'
print('WAV OK')
"
```

Listen to `/tmp/probe.wav`. If the Ukrainian is unintelligible, stop and try `tetiana/high` or `lada/x_low` instead, and record which voice was chosen.

- [ ] **Step 4: Measure the per-clip cost, including model load**

```bash
.venv/bin/python - <<'EOF'
import subprocess, time, pathlib
MODEL = 'models/tts_voices/uk_UA-ukrainian_tts-medium.onnx'
texts = ['Хрещатик', 'Ліве передсердя', 'Паспортні послуги', 'Софіївська вулиця', 'Сектор А']
for label, batch in (('single', texts[:1]), ('five separate processes', texts)):
    start = time.time()
    for i, t in enumerate(batch):
        subprocess.run(['.venv/bin/piper', '--model', MODEL,
                        '--output_file', f'/tmp/bench_{i}.wav'],
                       input=t, text=True, capture_output=True, check=True)
    elapsed = time.time() - start
    print(f'{label}: {elapsed:.2f}s total, {elapsed/len(batch):.2f}s per clip')
EOF
```

Record both numbers. This machine is faster than a Pi 4; the number that decides the design is measured in Task 11.

- [ ] **Step 5: Probe whether one process can take many clips**

```bash
printf '%s\n' \
  '{"text": "Хрещатик", "output_file": "/tmp/batch_a.wav"}' \
  '{"text": "Сектор А", "output_file": "/tmp/batch_b.wav"}' \
| .venv/bin/piper --model models/tts_voices/uk_UA-ukrainian_tts-medium.onnx --json-input
ls -la /tmp/batch_a.wav /tmp/batch_b.wav 2>&1
```

Record whether both files appeared. Task 3 deliberately uses one process per clip regardless; this only tells Task 11 whether batching is worth adding.

- [ ] **Step 6: Write down what was verified**

Create `docs/tts-setup.md` containing: the two install commands from Steps 1-2 verbatim, the chosen voice name, the Step 4 timings for this machine, and a yes/no for Step 5. Include the note that the aarch64 wheel covers a 64-bit Raspberry Pi OS and that a 32-bit OS needs a standalone piper binary instead.

- [ ] **Step 7: Commit**

```bash
git add docs/tts-setup.md
git commit -m "docs: record the verified Piper setup and synthesis timings"
```

---

### Task 2: pytest infrastructure and TTSConfig

**Files:**
- Create: `requirements-dev.txt`, `tests/conftest.py`
- Modify: `src/config.py` (append a new class), `.gitignore`

**Interfaces:**
- Consumes: the voice name verified in Task 1
- Produces: `TTSConfig.VOICE`, `.VOICES_DIR`, `.PIPER_BIN`, `.RUNTIME_FALLBACK`, `.GENERATED_SUBDIR`, `.TIMEOUT_SECONDS`; the `stub_piper` and `sample_model` fixtures used by Tasks 3-8

- [ ] **Step 1: Add the dev requirements and install them**

```bash
cat > requirements-dev.txt <<'EOF'
# Development-only dependencies. Runtime dependencies live in requirements.txt.
pytest>=8.0,<9.0
EOF
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest --version
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_tts_config.py`:

```python
"""TTSConfig carries the settings both the generator and the runtime fallback read."""

from src.config import TTSConfig


def test_voice_and_paths_are_configured():
    assert TTSConfig.VOICE == 'uk_UA-ukrainian_tts-medium'
    assert TTSConfig.VOICES_DIR == 'models/tts_voices'
    assert TTSConfig.GENERATED_SUBDIR == 'tts'


def test_runtime_fallback_is_on_by_default():
    assert TTSConfig.RUNTIME_FALLBACK is True


def test_piper_bin_defaults_to_path_lookup():
    assert TTSConfig.PIPER_BIN is None
```

If Task 1 Step 3 chose a different voice, use that name here instead.

- [ ] **Step 3: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_tts_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'TTSConfig'`

- [ ] **Step 4: Add TTSConfig**

Append to `src/config.py`:

```python
# ==================== Text-to-Speech Configuration ====================
class TTSConfig:
    """
    Configuration for generating zone narration with Piper.

    Used by both paths: the offline generator (src/tts/generate_audio.py) and the
    runtime fallback in ZoneAudioPlayer.
    """

    # Piper voice name. The voice is two files in VOICES_DIR:
    # <VOICE>.onnx and <VOICE>.onnx.json
    VOICE = 'uk_UA-ukrainian_tts-medium'

    # Where the downloaded voice lives. Not committed - the model is ~77 MB.
    # See docs/tts-setup.md for the download command.
    VOICES_DIR = 'models/tts_voices'

    # Path to the piper executable. None means "look it up on PATH".
    PIPER_BIN = None

    # Synthesize narration that is missing when a model is loaded. Turn this off
    # to run strictly on pre-generated files.
    RUNTIME_FALLBACK = True

    # Subdirectory of a model's audio directory where generated files are written.
    GENERATED_SUBDIR = 'tts'

    # Give up on a single clip after this long.
    TIMEOUT_SECONDS = 60
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tts_config.py -v`
Expected: 3 passed

- [ ] **Step 6: Add the shared fixtures**

Create `tests/conftest.py`:

```python
"""Fixtures shared by the TTS tests."""

import json
import os
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
    if 'BOOM' in text:
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
        '{\\n'
        '    "model": {\\n'
        '        "name": "Test",\\n'
        '        "map_description": "audio/Description.mp3",\\n'
        '        "mapDescriptionText": "Опис карти",\\n'
        '        "hotspots":[\\n'
        '        {"color":[255,0,0], "textDescription":"Хрещатик", '
        '"audioDescription":"audio/Khreschatyk.mp3"},\\n'
        '        {"color":[0,255,0], "textDescription":"Сектор А", '
        '"audioDescription":"audio/Sector.mp3"}\\n'
        '        ]\\n'
        '    }\\n'
        '}'
    )
    path = tmp_path / 'model.json'
    path.write_text(text, encoding='utf-8')
    return path
```

- [ ] **Step 7: Confirm the fixtures work**

Create `tests/test_conftest_fixtures.py`:

```python
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
```

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: 6 passed (the stub also writes <output_file>.txt; the WAV assertions are
unaffected)

- [ ] **Step 8: Keep the voice model out of git**

Append to `.gitignore`:

```
# Piper voice models - ~77 MB each, fetched per docs/tts-setup.md
models/tts_voices/
```

Verify: `git check-ignore -v models/tts_voices/uk_UA-ukrainian_tts-medium.onnx`

- [ ] **Step 9: Commit**

```bash
git add requirements-dev.txt tests/ src/config.py .gitignore
git commit -m "test: add pytest and the TTS fixtures, and TTSConfig"
```

---

### Task 3: The Piper engine

**Files:**
- Create: `src/tts/__init__.py`, `src/tts/engine.py`, `tests/test_tts_engine.py`

**Interfaces:**
- Consumes: `TTSConfig` from Task 2; the `stub_piper` and `stub_voice` fixtures
- Produces:
  - `engine.SynthesisJob(text: str, output_path: Path)` — frozen dataclass
  - `engine.TTSUnavailable` — exception
  - `engine.synthesize(jobs, *, piper_bin=None, voice=None, voices_dir=None, progress=None) -> tuple[list[SynthesisJob], list[SynthesisJob]]` returning `(succeeded, failed)`
  - `engine.resolve_piper(piper_bin=None) -> str`
  - `engine.resolve_voice(voice=None, voices_dir=None) -> tuple[Path, Path]`

One piper process per clip. That is certainly correct; batching is Task 11, and only if the Pi measurement justifies it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tts_engine.py`:

```python
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
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/bin/python -m pytest tests/test_tts_engine.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'src.tts'`

- [ ] **Step 3: Write the engine**

Create `src/tts/__init__.py`:

```python
"""Text-to-speech generation of zone narration."""
```

Create `src/tts/engine.py`:

```python
"""
Piper text-to-speech, wrapped as a subprocess.

The only module in the project that knows Piper exists; everything else asks it
for WAV files. Piper runs as a subprocess rather than through its Python API so
its ONNX runtime never shares a process with MediaPipe, and so a GPL-3.0
dependency stays at arm's length behind a process boundary.
"""

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from src.config import TTSConfig

logger = logging.getLogger(__name__)


class TTSUnavailable(RuntimeError):
    """Piper or its voice model cannot be used on this machine."""


@dataclass(frozen=True)
class SynthesisJob:
    """One piece of text to be written as one WAV file."""

    text: str
    output_path: Path


def _phonemizable(text):
    """
    Prepare text for piper by lowercasing it.

    The uk_UA voices' phoneme id maps have no entries for capital Cyrillic:
    'Хрещатик' logs "Missing phoneme from id map: Х" and is synthesized without
    its initial sound, while 'хрещатик' comes out whole. Every zone name starts
    with a capital, so this is not an edge case.

    It belongs here rather than in the narration text, because textDescription is
    also the zone name shown and logged elsewhere and has to stay capitalised.
    """
    return text.lower()


def resolve_piper(piper_bin=None):
    """
    Locate the piper executable.

    Checks, in order: the explicit argument, TTSConfig.PIPER_BIN, a `piper` sitting
    beside the running interpreter, then PATH.

    Args:
        piper_bin (str, optional): Explicit path. Falls back to TTSConfig, then to
            the interpreter's own directory, then PATH.

    Returns:
        str: Path to the executable.

    Raises:
        TTSUnavailable: Nothing usable was found.
    """
    candidate = piper_bin or TTSConfig.PIPER_BIN
    if candidate:
        if Path(candidate).is_file():
            return str(candidate)
        raise TTSUnavailable(f"piper not found at {candidate}")

    # Look beside the running interpreter before consulting PATH. piper installs
    # into .venv/bin/, and this project runs as .venv/bin/python without the venv
    # activated, so .venv/bin is not on PATH and shutil.which() misses it.
    beside_interpreter = Path(sys.executable).parent / 'piper'
    if beside_interpreter.is_file():
        return str(beside_interpreter)

    found = shutil.which('piper')
    if not found:
        raise TTSUnavailable(
            "piper is not on PATH. Install it with "
            "'uv pip install piper-tts' or set TTSConfig.PIPER_BIN. "
            "See docs/tts-setup.md."
        )
    return found


def resolve_voice(voice=None, voices_dir=None):
    """
    Locate a voice's two files.

    Returns:
        tuple[Path, Path]: The .onnx and .onnx.json paths.

    Raises:
        TTSUnavailable: Either file is missing, named in the message.
    """
    voice = voice or TTSConfig.VOICE
    directory = Path(voices_dir or TTSConfig.VOICES_DIR)
    onnx = directory / f'{voice}.onnx'
    config = directory / f'{voice}.onnx.json'

    missing = [str(p) for p in (onnx, config) if not p.is_file()]
    if missing:
        raise TTSUnavailable(
            f"voice files missing: {', '.join(missing)}. "
            f"See docs/tts-setup.md for the download command."
        )
    return onnx, config


def synthesize(jobs, *, piper_bin=None, voice=None, voices_dir=None, progress=None):
    """
    Write one WAV file per job, one piper process at a time.

    Args:
        jobs: Iterable of SynthesisJob.
        piper_bin (str, optional): Override the executable.
        voice (str, optional): Override the voice name.
        voices_dir (str, optional): Override the voice directory.
        progress (callable, optional): Called as progress(done, total, job) after
            each clip that succeeds.

    Returns:
        tuple[list, list]: (succeeded, failed) jobs. A clip that fails does not
            affect its neighbours.

    Raises:
        TTSUnavailable: piper or the voice could not be resolved, so nothing was
            attempted. Callers that must not crash should catch this.
    """
    jobs = list(jobs)
    if not jobs:
        return [], []

    piper = resolve_piper(piper_bin)
    onnx, _ = resolve_voice(voice, voices_dir)

    succeeded, failed = [], []
    for job in jobs:
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [piper, '--model', str(onnx), '--output_file', str(job.output_path)]
        try:
            result = subprocess.run(
                command,
                input=_phonemizable(job.text),
                text=True,
                capture_output=True,
                timeout=TTSConfig.TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            logger.error(f"piper failed for {job.output_path.name}: {e}")
            failed.append(job)
            continue

        if result.returncode != 0 or not job.output_path.is_file():
            detail = (result.stderr or '').strip()[:200]
            logger.error(
                f"piper failed for {job.output_path.name} "
                f"(exit {result.returncode}): {detail}"
            )
            failed.append(job)
            continue

        succeeded.append(job)
        if progress:
            progress(len(succeeded) + len(failed), len(jobs), job)

    return succeeded, failed
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tts_engine.py -v`
Expected: 10 passed

- [ ] **Step 5: Confirm it works against the real piper**

```bash
.venv/bin/python -c "
from pathlib import Path
from src.tts import engine
ok, failed = engine.synthesize(
    [engine.SynthesisJob(text='Хрещатик', output_path=Path('/tmp/real.wav'))])
print('ok:', [j.output_path.name for j in ok], 'failed:', failed)
"
```

Expected: `ok: ['real.wav'] failed: []`, and `/tmp/real.wav` is audible Ukrainian.

- [ ] **Step 6: Commit**

```bash
git add src/tts/__init__.py src/tts/engine.py tests/test_tts_engine.py
git commit -m "feat: add the Piper engine wrapper"
```

---

### Task 4: Narration planning

**Files:**
- Create: `src/tts/model_audio.py`, `tests/test_tts_model_audio.py`

**Interfaces:**
- Consumes: `TTSConfig.GENERATED_SUBDIR`
- Produces:
  - `model_audio.NarrationEntry(kind: str, index: int, text: str, audio_path: str)` — frozen dataclass; `kind` is `'hotspot'` or `'map_description'`, `index` is the hotspot position or `MAP_DESCRIPTION_INDEX` (-1)
  - `model_audio.MAP_DESCRIPTION_INDEX`
  - `model_audio.generated_path(audio_path, subdir=None) -> Path`
  - `model_audio.narration_entries(model: dict) -> list[NarrationEntry]`
  - `model_audio.pending(entries, *, force=False, subdir=None) -> list[NarrationEntry]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tts_model_audio.py`:

```python
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
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/bin/python -m pytest tests/test_tts_model_audio.py -v`
Expected: `ModuleNotFoundError: No module named 'src.tts.model_audio'`

- [ ] **Step 3: Write the module**

Create `src/tts/model_audio.py`:

```python
"""
Planning over a map model's narration entries.

Knows the shape of a model JSON and nothing else: no Piper, no audio backends, no
file formatting. That keeps it testable with a plain dict.
"""

from dataclasses import dataclass
from pathlib import Path

from src.config import TTSConfig

#: Stand-in index for the map description, which is not part of the hotspot list.
MAP_DESCRIPTION_INDEX = -1


@dataclass(frozen=True)
class NarrationEntry:
    """One piece of narration a model asks for."""

    kind: str          # 'hotspot' or 'map_description'
    index: int         # position in model['hotspots'], or MAP_DESCRIPTION_INDEX
    text: str          # what should be spoken; '' when the model has no text
    audio_path: str    # the path the model currently names

    @property
    def label(self):
        """Something human-readable for logs."""
        return self.text or f'{self.kind}[{self.index}]'


def generated_path(audio_path, subdir=None):
    """
    Where the generated WAV for an existing audio path belongs.

    The output goes into a subdirectory of whatever directory the current path
    points into, so a model that names its audio directory Sound/ keeps it:

        models/UkraineMap/Audio/Khreschyatik.mp3
            -> models/UkraineMap/Audio/tts/Khreschyatik.wav
        models/Heart/Sound/Aorta.mp3
            -> models/Heart/Sound/tts/Aorta.wav
    """
    path = Path(audio_path)
    subdir = subdir or TTSConfig.GENERATED_SUBDIR
    return path.parent / subdir / (path.stem + '.wav')


def narration_entries(model):
    """
    Every text-driven narration entry in a model, hotspots first.

    An entry with no text is still returned, so callers can warn about it instead
    of silently skipping it.
    """
    entries = [
        NarrationEntry(
            kind='hotspot',
            index=index,
            text=(hotspot.get('textDescription') or '').strip(),
            audio_path=hotspot.get('audioDescription', ''),
        )
        for index, hotspot in enumerate(model.get('hotspots', []))
    ]

    if model.get('map_description'):
        entries.append(NarrationEntry(
            kind='map_description',
            index=MAP_DESCRIPTION_INDEX,
            text=(model.get('mapDescriptionText') or '').strip(),
            audio_path=model['map_description'],
        ))

    return entries


def pending(entries, *, force=False, subdir=None):
    """
    The entries that need synthesizing.

    Skips anything with no text or no path. Skips anything whose output already
    exists unless force is set: existing audio is never replaced by accident.
    """
    result = []
    for entry in entries:
        if not entry.text or not entry.audio_path:
            continue
        if force or not generated_path(entry.audio_path, subdir).is_file():
            result.append(entry)
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tts_model_audio.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/tts/model_audio.py tests/test_tts_model_audio.py
git commit -m "feat: add narration planning over a model dict"
```

---

### Task 5: Formatting-preserving JSON edits

**Files:**
- Create: `src/tts/json_edit.py`, `tests/test_tts_json_edit.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `json_edit.JsonEditError` — exception
  - `json_edit.replace_string_value(text: str, key: str, old_value: str, new_value: str) -> str`
  - `json_edit.write_verified(path, text: str, expected_data) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tts_json_edit.py`:

```python
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
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/bin/python -m pytest tests/test_tts_json_edit.py -v`
Expected: `ModuleNotFoundError: No module named 'src.tts.json_edit'`

- [ ] **Step 3: Write the module**

Create `src/tts/json_edit.py`:

```python
"""
Targeted edits to a model JSON that leave its formatting alone.

The model files are hand-formatted with one hotspot per line, which keeps
UkraineMap.json at 34 readable lines where json.dump(indent=4) would produce 196.
The generator rewrites audioDescription on every run, so round-tripping through
the json module would bury one changed value in a 231-line diff every time.

Values are therefore replaced in the raw text, and the result is verified by
parsing it back and comparing against the data the caller expected. A textual
edit that produces invalid JSON, or the wrong data, is never written to disk.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class JsonEditError(RuntimeError):
    """A targeted edit could not be applied safely."""


def replace_string_value(text, key, old_value, new_value):
    """
    Replace one "key": "old_value" pair in raw JSON text.

    The key and both values are matched as JSON literals, so escaping and
    apostrophes inside values are handled by the json module rather than by a
    regex over hand-formatted text.

    Args:
        text (str): The whole file's contents.
        key (str): The object key whose value should change.
        old_value (str): The current value, matched exactly.
        new_value (str): The replacement.

    Returns:
        str: The edited text.

    Raises:
        JsonEditError: The pair was absent, or present more than once and
            therefore ambiguous.
    """
    key_literal = json.dumps(key, ensure_ascii=False)
    old_literal = json.dumps(old_value, ensure_ascii=False)
    new_literal = json.dumps(new_value, ensure_ascii=False)

    # The models use both "key":"value" and "key": "value".
    for gap in ('', ' '):
        needle = f'{key_literal}:{gap}{old_literal}'
        occurrences = text.count(needle)
        if occurrences == 1:
            return text.replace(needle, f'{key_literal}:{gap}{new_literal}', 1)
        if occurrences > 1:
            raise JsonEditError(
                f'{key}={old_value!r} appears {occurrences} times; '
                f'cannot edit it unambiguously'
            )

    raise JsonEditError(f'could not find {key}={old_value!r} in the text')


def write_verified(path, text, expected_data):
    """
    Write edited JSON text, but only once it is proven correct.

    Args:
        path: Destination file.
        text (str): The edited text.
        expected_data: What json.loads(text) must equal.

    Raises:
        JsonEditError: The text does not parse, or parses to something other than
            expected_data. Nothing is written in either case.
    """
    try:
        actual = json.loads(text)
    except json.JSONDecodeError as e:
        raise JsonEditError(f'edited text is not valid JSON: {e}') from e

    if actual != expected_data:
        raise JsonEditError(
            'edited text does not contain the expected data; refusing to write'
        )

    Path(path).write_text(text, encoding='utf-8')
    logger.debug(f'wrote {path}')
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tts_json_edit.py -v`
Expected: 9 passed

- [ ] **Step 5: Prove it against a real model file**

```bash
.venv/bin/python - <<'EOF'
import json
from src.tts import json_edit

path = 'models/UkraineMap/UkraineMap.json'
raw = open(path, encoding='utf-8').read()
edited = json_edit.replace_string_value(
    raw, 'audioDescription',
    'models/UkraineMap/Audio/Khreschyatik.mp3',
    'models/UkraineMap/Audio/tts/Khreschyatik.wav')
before, after = raw.split('\n'), edited.split('\n')
changed = [i for i, (a, b) in enumerate(zip(before, after), 1) if a != b]
print(f'lines before {len(before)}, after {len(after)}, changed lines: {changed}')
assert len(before) == len(after) and len(changed) == 1, 'formatting was not preserved'
print('formatting preserved')
EOF
```

Expected: one changed line, same total. Nothing is written to disk by this probe.

- [ ] **Step 6: Commit**

```bash
git add src/tts/json_edit.py tests/test_tts_json_edit.py
git commit -m "feat: add verified, formatting-preserving JSON edits"
```

---

### Task 6: The generator CLI

**Files:**
- Create: `src/tts/generate_audio.py`, `tests/test_tts_generate_audio.py`

**Interfaces:**
- Consumes: `engine.SynthesisJob`, `engine.synthesize`, `engine.TTSUnavailable`, `model_audio.narration_entries`, `model_audio.pending`, `model_audio.generated_path`, `json_edit.replace_string_value`, `json_edit.write_verified`
- Produces: `generate_audio.generate_for_model(model_path, *, force=False, voice=None, voices_dir=None, piper_bin=None) -> int` returning the number of failed entries; `generate_audio.main(argv=None) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tts_generate_audio.py`:

```python
"""End-to-end over the CLI's function, with the stub piper standing in."""

import json

import pytest

from src.tts import generate_audio


def _run(model_file, stub_piper, stub_voice, **kwargs):
    return generate_audio.generate_for_model(
        model_file, piper_bin=str(stub_piper), voice='test_voice',
        voices_dir=stub_voice, **kwargs)


def test_generates_every_missing_clip(sample_model_file, stub_piper, stub_voice):
    failed = _run(sample_model_file, stub_piper, stub_voice)
    assert failed == 0
    audio = sample_model_file.parent / 'audio' / 'tts'
    assert sorted(p.name for p in audio.iterdir()) == \
        ['Description.wav', 'Khreschatyk.wav', 'Sector.wav']


def test_rewrites_only_the_paths_it_generated(sample_model_file, stub_piper, stub_voice):
    _run(sample_model_file, stub_piper, stub_voice)
    model = json.loads(sample_model_file.read_text(encoding='utf-8'))['model']
    assert model['hotspots'][0]['audioDescription'] == 'audio/tts/Khreschatyk.wav'
    assert model['hotspots'][1]['audioDescription'] == 'audio/tts/Sector.wav'
    assert model['map_description'] == 'audio/tts/Description.wav'


def test_keeps_the_files_formatting(sample_model_file, stub_piper, stub_voice):
    before = sample_model_file.read_text(encoding='utf-8')
    _run(sample_model_file, stub_piper, stub_voice)
    after = sample_model_file.read_text(encoding='utf-8')
    assert len(before.split('\n')) == len(after.split('\n'))
    assert not after.endswith('\n')
    assert '\\u0425' not in after


def test_a_second_run_generates_nothing(sample_model_file, stub_piper, stub_voice):
    _run(sample_model_file, stub_piper, stub_voice)
    after_first = sample_model_file.read_text(encoding='utf-8')
    stamps = {p: p.stat().st_mtime_ns
              for p in (sample_model_file.parent / 'audio' / 'tts').iterdir()}

    assert _run(sample_model_file, stub_piper, stub_voice) == 0
    assert sample_model_file.read_text(encoding='utf-8') == after_first
    assert {p: p.stat().st_mtime_ns for p in stamps} == stamps


def test_force_regenerates(sample_model_file, stub_piper, stub_voice):
    _run(sample_model_file, stub_piper, stub_voice)
    target = sample_model_file.parent / 'audio' / 'tts' / 'Khreschatyk.wav'
    target.write_bytes(b'clobbered')
    _run(sample_model_file, stub_piper, stub_voice, force=True)
    assert target.read_bytes() != b'clobbered'


def test_a_failing_clip_is_counted_and_its_path_is_not_rewritten(
        tmp_path, stub_piper, stub_voice):
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
        tmp_path, stub_piper, stub_voice):
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
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/bin/python -m pytest tests/test_tts_generate_audio.py -v`
Expected: `ModuleNotFoundError: No module named 'src.tts.generate_audio'`

- [ ] **Step 3: Write the CLI**

Create `src/tts/generate_audio.py`:

```python
"""
Generate the spoken narration for a map model.

    python -m src.tts.generate_audio --input1 models/UkraineMap/UkraineMap.json

Only missing audio is generated, and audioDescription is rewritten just for the
entries that were produced. Existing audio is never replaced without --force.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.tts import engine, json_edit, model_audio

logger = logging.getLogger(__name__)


def generate_for_model(model_path, *, force=False, voice=None, voices_dir=None,
                       piper_bin=None):
    """
    Synthesize whatever the model is missing and point it at the results.

    Args:
        model_path: Path to the model JSON.
        force (bool): Regenerate audio that already exists.
        voice, voices_dir, piper_bin: Overrides passed through to the engine.

    Returns:
        int: How many entries failed. 0 means everything asked for was produced.
    """
    model_path = Path(model_path)
    raw = model_path.read_text(encoding='utf-8')
    document = json.loads(raw)
    model = document['model']

    entries = model_audio.narration_entries(model)
    for entry in entries:
        if not entry.text:
            logger.warning(
                f'no text for {entry.kind}[{entry.index}] '
                f'({entry.audio_path or "no path"}) - skipping'
            )

    todo = model_audio.pending(entries, force=force)
    if not todo:
        logger.info('nothing to generate')
        return 0

    jobs = [
        engine.SynthesisJob(
            text=entry.text,
            output_path=model_audio.generated_path(entry.audio_path),
        )
        for entry in todo
    ]

    def report(done, total, job):
        logger.info(f'synthesized {done}/{total}: {job.output_path.name}')

    try:
        succeeded, failed = engine.synthesize(
            jobs, piper_bin=piper_bin, voice=voice, voices_dir=voices_dir,
            progress=report,
        )
    except engine.TTSUnavailable as e:
        logger.error(str(e))
        return len(jobs)

    produced = {job.output_path for job in succeeded}

    text = raw
    expected = json.loads(raw)
    for entry, job in zip(todo, jobs):
        if job.output_path not in produced:
            continue
        new_path = str(job.output_path)
        if new_path == entry.audio_path:
            continue
        key = 'audioDescription' if entry.kind == 'hotspot' else 'map_description'
        text = json_edit.replace_string_value(text, key, entry.audio_path, new_path)
        if entry.kind == 'hotspot':
            expected['model']['hotspots'][entry.index]['audioDescription'] = new_path
        else:
            expected['model']['map_description'] = new_path

    if text != raw:
        json_edit.write_verified(model_path, text, expected)
        logger.info(f'updated audio paths in {model_path}')

    for job in failed:
        logger.error(f'failed: {job.output_path.name}')

    return len(failed)


def main(argv=None):
    """Entry point. Returns the process exit code."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    parser = argparse.ArgumentParser(
        description='Generate spoken narration for a map model with Piper')
    parser.add_argument('--input1', required=True, metavar='MODEL_JSON',
                        help='Path to the map configuration JSON')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate audio that already exists')
    parser.add_argument('--voice', default=None,
                        help='Override TTSConfig.VOICE')
    parser.add_argument('--voices-dir', default=None,
                        help='Override TTSConfig.VOICES_DIR')
    parser.add_argument('--piper-bin', default=None,
                        help='Override TTSConfig.PIPER_BIN')
    args = parser.parse_args(argv)

    failed = generate_for_model(
        args.input1, force=args.force, voice=args.voice,
        voices_dir=args.voices_dir, piper_bin=args.piper_bin,
    )
    if failed:
        logger.error(f'{failed} entr{"y" if failed == 1 else "ies"} failed')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tts_generate_audio.py -v`
Expected: 8 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all pass, nothing broken by the new modules

- [ ] **Step 6: Commit**

```bash
git add src/tts/generate_audio.py tests/test_tts_generate_audio.py
git commit -m "feat: add the generate_audio CLI"
```

---

### Task 7: One place to load a sound

A refactor with one behaviour change: a missing `map_description` no longer raises.

**Files:**
- Modify: `src/audio/audio.py`
- Create: `tests/test_audio_loading.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `ZoneAudioPlayer._load_sound(path, label=None)` returning a backend sound object or `None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_audio_loading.py`:

```python
"""
_load_sound is the single place that turns a path into a playable sound, so it is
also the single place that has to survive a missing file.
"""

import src.audio.audio as audio_module
from src.audio.audio import ZoneAudioPlayer


def _player_without_init():
    """A ZoneAudioPlayer instance with no __init__ run - we only want the method."""
    return ZoneAudioPlayer.__new__(ZoneAudioPlayer)


def test_missing_file_returns_none_and_does_not_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_module, 'USE_PYGLET', False)
    monkeypatch.setattr(audio_module, 'USE_PYGAME', False)
    assert _player_without_init()._load_sound(str(tmp_path / 'nope.wav')) is None


def test_empty_path_returns_none(monkeypatch):
    monkeypatch.setattr(audio_module, 'USE_PYGLET', False)
    monkeypatch.setattr(audio_module, 'USE_PYGAME', False)
    assert _player_without_init()._load_sound('') is None


def test_no_backend_returns_none_for_a_file_that_exists(monkeypatch, tmp_path):
    real = tmp_path / 'real.wav'
    real.write_bytes(b'RIFF')
    monkeypatch.setattr(audio_module, 'USE_PYGLET', False)
    monkeypatch.setattr(audio_module, 'USE_PYGAME', False)
    assert _player_without_init()._load_sound(str(real)) is None


def test_a_backend_that_raises_is_contained(monkeypatch, tmp_path):
    """A corrupt clip must not take the application down."""
    real = tmp_path / 'real.wav'
    real.write_bytes(b'not really a wav')

    class Boom:
        @staticmethod
        def load(path, streaming=False):
            raise RuntimeError('decoder blew up')

    monkeypatch.setattr(audio_module, 'USE_PYGLET', True)
    monkeypatch.setattr(audio_module, 'USE_PYGAME', False)
    monkeypatch.setitem(__import__('sys').modules, 'pyglet.media', Boom)
    assert _player_without_init()._load_sound(str(real)) is None
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/bin/python -m pytest tests/test_audio_loading.py -v`
Expected: FAIL, `AttributeError: 'ZoneAudioPlayer' object has no attribute '_load_sound'`

- [ ] **Step 3: Add the helper**

Add to `ZoneAudioPlayer` in `src/audio/audio.py`, immediately above `_load_hotspot_audio`:

```python
    def _load_sound(self, path, label=None):
        """
        Load one audio file with whichever backend is active.

        Args:
            path (str): Path to the audio file.
            label (str, optional): Name to use in log messages.

        Returns:
            The backend's sound object, or None when the path is empty, the file is
            absent, or no backend is available. Callers must tolerate None: one
            silent clip is survivable where an exception would end the session.
        """
        if not path:
            return None

        label = label or path
        if not os.path.exists(path):
            logger.warning(f"Audio file not found: {path}")
            return None

        try:
            if USE_PYGLET:
                import pyglet.media
                return pyglet.media.load(path, streaming=False)
            if USE_PYGAME:
                import pygame
                return pygame.mixer.Sound(path)
        except Exception as e:
            logger.error(f"Could not load {label}: {e}")
        return None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_audio_loading.py -v`
Expected: 4 passed

- [ ] **Step 5: Route the five consumers through it**

In `ZoneAudioPlayer.__init__`, replace exactly the `if USE_PYGLET: / elif USE_PYGAME: / else:`
block that loads `blipsound`, `map_description`, `welcome_message` and
`goodbye_message`. It starts at the `if USE_PYGLET:` line following
`self.enable_blips = False`, and ends at the `self.have_played_description = True`
inside the final `else:`. Stop before the `# Load audio files for each hotspot`
comment and the `self._load_hotspot_audio()` call, and leave everything above the
block alone — `self.model`, `self.prev_zone_name`, `self.prev_zone_moving`,
`self.curr_zone_moving`, `self.sound_files`, `self.hotspots` and
`self.enable_blips` are assigned there and are all still needed. Replace it with:

```python
        if USE_PYGLET:
            import pyglet.media
            self.player = pyglet.media.Player()
        else:
            self.player = None
        self.welcome_player = None
        self.goodbye_player = None
        self.current_channel = None

        self.blip_sound = self._load_sound(self.model.get('blipsound'), 'blip sound')
        self.map_description = self._load_sound(
            self.model.get('map_description'), 'map description')
        self.have_played_description = self.map_description is None
        self.welcome_message = self._load_sound(
            self.model.get('welcome_message'), 'welcome message')
        self.goodbye_message = self._load_sound(
            self.model.get('goodbye_message'), 'goodbye message')
```

`have_played_description` starts True when there is no description, which is what the old code did for a model with no `map_description` key. The difference is that a key pointing at an absent file no longer raises.

In `_load_hotspot_audio`, replace the body of the loop after `self.hotspots[key] = hotspot` with:

```python
            sound = self._load_sound(
                hotspot.get('audioDescription'), hotspot.get('textDescription'))
            if sound is not None:
                self.sound_files[key] = sound
```

- [ ] **Step 6: Verify nothing regressed**

```bash
.venv/bin/python -m pytest tests/ -v
.venv/bin/python -m compileall -q src && echo "compiles clean"
timeout 40 .venv/bin/python simple_camio.py --headless --camera 0 \
  --input1 models/UkraineMap/UkraineMap.json < /dev/null > /tmp/t7.log 2>&1 &
P=$!; sleep 10; kill -TERM $P; wait $P 2>/dev/null
grep -E 'zone audio player|Cleanup complete' /tmp/t7.log
grep -c Traceback /tmp/t7.log
```

Expected: tests pass, `Initialized zone audio player (...) with 18 hotspots`, `Cleanup complete`, 0 tracebacks.

- [ ] **Step 7: Prove the map_description crash is gone**

```bash
.venv/bin/python - <<'EOF'
import json, shutil, tempfile, os
src = 'models/UkraineMap/UkraineMap.json'
tmp = tempfile.mkdtemp()
doc = json.loads(open(src, encoding='utf-8').read())
doc['model']['map_description'] = 'models/UkraineMap/Audio/does-not-exist.mp3'
target = os.path.join(tmp, 'broken.json')
open(target, 'w', encoding='utf-8').write(json.dumps(doc, ensure_ascii=False))
from src.audio.audio import ZoneAudioPlayer
player = ZoneAudioPlayer(doc['model'])
print('constructed with a missing map description; have_played_description =',
      player.have_played_description)
EOF
```

Expected: it constructs and prints `True`. Before this task it raised.

- [ ] **Step 8: Commit**

```bash
git add src/audio/audio.py tests/test_audio_loading.py
git commit -m "refactor: load every clip through one helper that tolerates a missing file"
```

---

### Task 8: The runtime fallback

**Files:**
- Modify: `src/audio/audio.py`
- Create: `tests/test_audio_fallback.py`

**Interfaces:**
- Consumes: `engine.SynthesisJob`, `engine.synthesize`, `engine.TTSUnavailable`, `model_audio.narration_entries`, `TTSConfig.RUNTIME_FALLBACK`
- Produces: `ZoneAudioPlayer._synthesize_missing()`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_audio_fallback.py`:

```python
"""
The fallback runs once at load, so a finger resting on a zone never waits for
synthesis, and a broken TTS setup never takes the player down with it.
"""

import src.audio.audio as audio_module
from src.audio.audio import ZoneAudioPlayer
from src.tts import engine


def _player(model):
    player = ZoneAudioPlayer.__new__(ZoneAudioPlayer)
    player.model = model
    return player


def _model(tmp_path):
    return {
        'hotspots': [
            {'textDescription': 'Хрещатик',
             'audioDescription': str(tmp_path / 'a.wav')},
            {'textDescription': 'Сектор А',
             'audioDescription': str(tmp_path / 'b.wav')},
        ],
    }


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
    (tmp_path / 'a.wav').write_bytes(b'RIFF')
    calls = []

    def fake_synthesize(jobs, **kwargs):
        jobs = list(jobs)
        calls.append(jobs)
        return jobs, []

    monkeypatch.setattr(engine, 'synthesize', fake_synthesize)
    _player(_model(tmp_path))._synthesize_missing()
    assert [j.output_path.name for j in calls[0]] == ['b.wav']


def test_nothing_missing_means_no_engine_call(monkeypatch, tmp_path):
    (tmp_path / 'a.wav').write_bytes(b'RIFF')
    (tmp_path / 'b.wav').write_bytes(b'RIFF')

    def explode(jobs, **kwargs):
        raise AssertionError('the engine must not be called')

    monkeypatch.setattr(engine, 'synthesize', explode)
    _player(_model(tmp_path))._synthesize_missing()


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
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/bin/python -m pytest tests/test_audio_fallback.py -v`
Expected: FAIL, `AttributeError: ... has no attribute '_synthesize_missing'`

- [ ] **Step 3: Add the imports**

At the top of `src/audio/audio.py`, beside the existing `import os`:

```python
from pathlib import Path

from src.config import TTSConfig
```

- [ ] **Step 4: Add the method**

Add to `ZoneAudioPlayer`, immediately above `_load_sound`:

```python
    def _synthesize_missing(self):
        """
        Fill in narration audio that is absent, before anything is loaded.

        Runs once at construction rather than on demand, so a zone never waits for
        synthesis while a finger is resting on it. Writes only audio files, at the
        paths the model already names - a device in the field does not rewrite its
        own configuration.

        Every failure here is logged and tolerated. The player continues with
        whatever files do exist.
        """
        if not TTSConfig.RUNTIME_FALLBACK:
            return

        # Lazy: Piper may not be installed, and that must not stop the app starting.
        from src.tts import engine, model_audio

        pending = [
            entry for entry in model_audio.narration_entries(self.model)
            if entry.text and entry.audio_path and not os.path.exists(entry.audio_path)
        ]
        if not pending:
            return

        logger.info(f"{len(pending)} narration clip(s) missing - synthesizing")
        jobs = [
            engine.SynthesisJob(text=entry.text, output_path=Path(entry.audio_path))
            for entry in pending
        ]

        def report(done, total, job):
            logger.info(f"synthesizing {done}/{total}: {job.output_path.name}")

        try:
            _, failed = engine.synthesize(jobs, progress=report)
        except engine.TTSUnavailable as e:
            logger.warning(f"Cannot synthesize the missing narration: {e}")
            return
        except Exception as e:
            logger.error(f"Synthesis failed unexpectedly: {e}", exc_info=True)
            return

        for job in failed:
            logger.warning(
                f"No audio for {job.output_path.name}; that zone will stay silent")
```

- [ ] **Step 5: Call it before anything loads**

In `ZoneAudioPlayer.__init__`, insert the call after the plain attribute
assignments and immediately before the first `self.blip_sound = self._load_sound(...)`
line from Task 7, so synthesis finishes before anything is loaded:

```python
        self._synthesize_missing()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all pass

- [ ] **Step 7: Prove the fallback works against real piper**

```bash
.venv/bin/python - <<'EOF'
import json, os, tempfile
from src.audio.audio import ZoneAudioPlayer

tmp = tempfile.mkdtemp()
model = {
    'blipsound': 'MP3/quick_blip.wav',
    'welcome_message': 'MP3/welcome.mp3',
    'goodbye_message': 'MP3/goodbye.mp3',
    'hotspots': [
        {'color': [255, 0, 0], 'textDescription': 'Хрещатик',
         'audioDescription': os.path.join(tmp, 'khreschatyk.wav')},
    ],
}
player = ZoneAudioPlayer(model)
generated = os.path.join(tmp, 'khreschatyk.wav')
print('generated:', os.path.exists(generated), os.path.getsize(generated), 'bytes')
print('loaded into the player:', len(player.sound_files), 'clip(s)')
EOF
```

Expected: the file exists, is more than a few kB, and one clip is loaded. Listen to it.

- [ ] **Step 8: Confirm a missing piper is survivable**

```bash
.venv/bin/python - <<'EOF'
import os, tempfile
from src.config import TTSConfig
TTSConfig.PIPER_BIN = '/nonexistent/piper'
from src.audio.audio import ZoneAudioPlayer
tmp = tempfile.mkdtemp()
player = ZoneAudioPlayer({
    'blipsound': 'MP3/quick_blip.wav',
    'welcome_message': 'MP3/welcome.mp3',
    'goodbye_message': 'MP3/goodbye.mp3',
    'hotspots': [{'color': [255, 0, 0], 'textDescription': 'Хрещатик',
                  'audioDescription': os.path.join(tmp, 'x.wav')}],
})
print('player constructed anyway, clips loaded:', len(player.sound_files))
EOF
```

Expected: a warning naming the missing piper, then `player constructed anyway, clips loaded: 0`.

- [ ] **Step 9: Commit**

```bash
git add src/audio/audio.py tests/test_audio_fallback.py
git commit -m "feat: synthesize missing narration when a model loads"
```

---

### Task 9: Rewrite the narration text in Cyrillic

Content, not code. It lands on its own so the JSON diff is the review surface.

**Files:**
- Modify: `models/UkraineMap/UkraineMap.json`, `models/Heart/Heart.json`, `models/CnapMap/CnapFirstFloor.json`
- Create: `tests/test_model_text.py`

**Interfaces:**
- Consumes: `json_edit.replace_string_value`, `json_edit.write_verified`
- Produces: Cyrillic `textDescription` on all 52 hotspots, and `mapDescriptionText` on all three models

- [ ] **Step 1: Dump what is there now**

```bash
.venv/bin/python - <<'EOF'
import json, glob
for path in sorted(glob.glob('models/*/*.json')):
    document = json.loads(open(path, encoding='utf-8').read())
    if 'model' not in document:
        continue          # e.g. models/tts_voices/*.onnx.json - a voice, not a map
    model = document['model']
    print(f'\n=== {path} ===')
    for i, h in enumerate(model.get('hotspots', [])):
        print(f'  [{i:2d}] {h.get("textDescription", "")}')
    print(f'  map_description audio: {model.get("map_description", "")}')
EOF
```

- [ ] **Step 2: Draft the Cyrillic column**

For Heart and UkraineMap this is de-transliteration. Use these:

**Heart** — `Aorta` → `Аорта`; `Live peredserdya` → `Ліве передсердя`; `Liviy shlunochok` → `Лівий шлуночок`; `Miokard` → `Міокард`; `Prave peredserdya` → `Праве передсердя`; `Praviy shlunochok` → `Правий шлуночок`. Take the seventh from the Step 1 dump and de-transliterate it the same way.

**UkraineMap** — `Khreschatyk St.` → `Хрещатик`; `Mykhailivs'ka St.` → `Михайлівська вулиця`; `Mala Zhytomyrska St.` → `Мала Житомирська вулиця`; `Sofiivs'ka St.` → `Софіївська вулиця`; `Tarasa Shevchenka Ln.` → `Провулок Тараса Шевченка`; `Borysa Hrinchenka St.` → `Вулиця Бориса Грінченка`. Continue from the dump for the remaining twelve, keeping the pattern: bare name for `Хрещатик` and `Майдан`, `вулиця`/`провулок`/`узвіз` spelled out otherwise.

**CnapMap** — mixed. Straightforward service names: `Passport Services` → `Паспортні послуги`; `Consultations` → `Консультації`; `Residence registration services` → `Реєстрація місця проживання`; `Stairs to the top` → `Сходи вгору`; `Stairs to the bottom` → `Сходи вниз`. Labels like `Sector A` are probably printed on physical signage; draft `Сектор А` but mark every one of them for the owner to confirm against the real sign.

Write the full 52-row draft into the pull request or chat as a three-column table (index, current, proposed) grouped by model, with the signage-dependent CnapMap rows flagged.

- [ ] **Step 3: Note that the approval gate was waived**

The owner waived the review gate for this run: apply the drafted table and let the
JSON diff serve as the review surface after the fact. Record in the commit message
that the Cyrillic is unreviewed. The maps still have printed physical counterparts,
and the CnapMap labels in particular have to match the real signage, so the diff
needs a human pass before this reaches a deployed device.

- [ ] **Step 4: Apply the approved text**

Write a one-off script that uses `json_edit.replace_string_value` per value and `json_edit.write_verified` per file, so formatting survives:

```python
"""Apply the approved Cyrillic narration text. Run once, then delete."""

import json
from pathlib import Path

from src.tts import json_edit

# Transcribe the table approved in Step 3 into these two dicts verbatim. Do not add,
# drop or reword an entry: the approved table is the source of truth, and anything
# not in it has not been checked against the printed map.
#
# path -> {current textDescription: approved Cyrillic}, 52 rows in total
TEXT = {
    'models/Heart/Heart.json': {'Aorta': 'Аорта'},
    'models/UkraineMap/UkraineMap.json': {},
    'models/CnapMap/CnapFirstFloor.json': {},
}

# path -> the approved mapDescriptionText, one per model
DESCRIPTIONS = {
    'models/Heart/Heart.json': '',
    'models/UkraineMap/UkraineMap.json': '',
    'models/CnapMap/CnapFirstFloor.json': '',
}

for path, mapping in TEXT.items():
    raw = Path(path).read_text(encoding='utf-8')
    text = raw
    expected = json.loads(raw)
    for old, new in mapping.items():
        text = json_edit.replace_string_value(text, 'textDescription', old, new)
    for hotspot in expected['model']['hotspots']:
        current = hotspot.get('textDescription')
        if current in mapping:
            hotspot['textDescription'] = mapping[current]
    json_edit.write_verified(path, text, expected)
    print(f'{path}: {len(mapping)} values rewritten')
```

`mapDescriptionText` is a new key, so it cannot be replaced — insert it with a targeted edit immediately after the `"map_description"` line, matching the surrounding indentation, then verify by parsing.

- [ ] **Step 5: Write the guard test**

Create `tests/test_model_text.py`:

```python
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
```

- [ ] **Step 6: Verify the diff is the review surface**

```bash
.venv/bin/python -m pytest tests/test_model_text.py -v
git diff --stat models/
git diff models/UkraineMap/UkraineMap.json | head -30
```

Expected: tests pass; the diff shows one changed line per rewritten value, not a reformatted file. If a file's line count moved by more than the number of inserted `mapDescriptionText` keys, the formatting was not preserved — stop and fix it.

- [ ] **Step 7: Confirm the app still names zones correctly**

`textDescription` also feeds the zone name at `simple_camio.py:267`, `simple_camio.py:306` and `src/audio/audio.py:414`, so those log lines become Ukrainian.

```bash
timeout 40 .venv/bin/python simple_camio.py --headless --camera 0 \
  --input1 models/UkraineMap/UkraineMap.json < /dev/null > /tmp/t9.log 2>&1 &
P=$!; sleep 10; kill -TERM $P; wait $P 2>/dev/null
grep -c Traceback /tmp/t9.log
grep -E 'Cleanup complete' /tmp/t9.log
```

Expected: 0 tracebacks, clean shutdown.

- [ ] **Step 8: Commit, content only**

```bash
git add models/ tests/test_model_text.py
git commit -m "content: rewrite the narration text in Ukrainian"
```

---

### Task 10: Generate the real audio and document it

**Files:**
- Modify: `models/*/*.json` (audio paths), `README.md`, `ARCHITECTURE.md`, `requirements.txt`
- Untracked output: `models/*/{Audio,Sound}/tts/*.wav`

**Interfaces:**
- Consumes: everything above
- Produces: generated narration for all three maps and the documentation to reproduce it

- [ ] **Step 1: Keep the generated audio out of git**

Decided by the owner: the WAVs are build output from text that is already in git, so
they are not committed. That keeps the repository small and makes the runtime
fallback the mechanism that actually matters. Add to `.gitignore`:

```
# Generated narration - rebuild with python -m src.tts.generate_audio
models/*/Audio/tts/
models/*/Sound/tts/
```

- [ ] **Step 2: Generate all three maps**

```bash
for m in models/UkraineMap/UkraineMap.json \
         models/CnapMap/CnapFirstFloor.json \
         models/Heart/Heart.json; do
  echo "=== $m ==="
  .venv/bin/python -m src.tts.generate_audio --input1 "$m"
done
```

Expected: exit 0 for each, and 53 WAV files in total across the three `tts/` directories.

- [ ] **Step 3: Listen to a sample from each map**

Play one clip per model. A generated file that is silent, truncated, or reads the text as English is a failure even though the command exited 0.

- [ ] **Step 4: Run the app on generated audio only**

```bash
timeout 40 .venv/bin/python simple_camio.py --headless --camera 0 \
  --input1 models/UkraineMap/UkraineMap.json < /dev/null > /tmp/t10.log 2>&1 &
P=$!; sleep 10; kill -TERM $P; wait $P 2>/dev/null
grep -E 'zone audio player|narration clip|Cleanup complete' /tmp/t10.log
grep -c Traceback /tmp/t10.log
```

Expected: 18 hotspots loaded, **no** "narration clip(s) missing" line — everything came from disk — and 0 tracebacks.

- [ ] **Step 5: Add piper to the runtime requirements**

Append to `requirements.txt`:

```
# Text-to-speech for zone narration. GPL-3.0, run as a subprocess - see
# src/tts/engine.py. Only needed to generate audio, or for the runtime fallback
# when a map ships without it. Voice models are fetched separately; see
# docs/tts-setup.md.
piper-tts>=1.7.0,<2.0
```

- [ ] **Step 6: Document it**

In `README.md`, after the "Configuration" section, add a "Zone narration" section covering: that zone audio is generated from each hotspot's `textDescription`, the `generate_audio` command, the voice download pointer to `docs/tts-setup.md`, and that a map with missing clips synthesizes them at startup instead of failing.

In `ARCHITECTURE.md`, add `src/tts/` to the project tree with its four modules, and a "Zone narration" subsection under "Module Descriptions" describing the two paths and why Piper is a subprocess.

- [ ] **Step 7: Verify the documented commands work verbatim**

Copy each command out of the new documentation and run it. A documented command that does not run is the defect this project has already been through once.

- [ ] **Step 8: Commit**

```bash
git add models/ README.md ARCHITECTURE.md requirements.txt .gitignore
git commit -m "feat: generate the zone narration and document the workflow"
```

---

### Task 11: Verify on the Raspberry Pi 4

Requires the physical device. An agent cannot complete this task; it produces the measurement that decides whether the load-time fallback survives as designed.

**Files:**
- Modify: `docs/tts-setup.md`

- [ ] **Step 1: Confirm the architecture**

```bash
uname -m
```

`aarch64` means the piper wheel installs directly. `armv7l` means a 32-bit OS: the wheel does not apply, and a standalone piper binary or a source build is needed. Record which.

- [ ] **Step 2: Install piper and the voice on the Pi**

Follow `docs/tts-setup.md` exactly. If a step does not work on the Pi, fix the document.

- [ ] **Step 3: Measure one clip**

```bash
time (echo 'Хрещатик' | piper \
  --model models/tts_voices/uk_UA-ukrainian_tts-medium.onnx \
  --output_file /tmp/pi.wav)
```

Record the wall time, including model load.

- [ ] **Step 4: Measure the worst case honestly**

CnapMap has 27 zones. Multiply the Step 3 time by 27 — that is how long a first startup is silent if a map arrives without audio.

Decide against this number:

- **Under about 15 s total** — the design stands, no change needed.
- **15 s to a minute** — keep load-time synthesis but log progress prominently, and treat pre-generation as mandatory for deployment rather than merely recommended.
- **Over a minute** — the load-time fallback is not a fallback. Add batching first (one piper process for all clips, which the Task 1 Step 5 probe already told us is possible or not) and measure again. If it is still over a minute, revisit the spec's fallback-timing decision with the owner: background synthesis after startup becomes the better trade.

- [ ] **Step 5: Run the app on the Pi under systemd**

```bash
sudo systemctl restart simple_camio
journalctl -u simple_camio -f
```

Expected: `Audio backend: pygame` or `pyglet`, zone audio loaded, no tracebacks, and narration audible when a finger dwells on a zone.

- [ ] **Step 6: Judge the voice**

Have someone who speaks Ukrainian listen to several zones on the physical map. Street names, room names, and anything with an unusual stress pattern are where a TTS voice fails. If the voice is not good enough, try `uk_UA-tetiana-high` (better, slower) or `uk_UA-lada-x_low` (faster, worse) and re-run Steps 3-4.

- [ ] **Step 7: Record the results and commit**

Add a "Measured on Raspberry Pi 4" section to `docs/tts-setup.md` with the architecture, the per-clip time, the 27-zone projection, the decision taken at Step 4, and the voice chosen.

```bash
git add docs/tts-setup.md
git commit -m "docs: record the Raspberry Pi 4 synthesis measurements"
```
