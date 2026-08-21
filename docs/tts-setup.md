# Piper TTS setup (verified)

This records what was actually run and observed while verifying the Piper
toolchain for zone-narration audio generation. All commands were run from the
repository root, on x86_64 Ubuntu 26.04 (not the target Raspberry Pi), using
`.venv/bin/python` / `.venv/bin/piper`.

## Install

piper-tts is not in `requirements.txt` - it has no wheel for 32-bit
Raspberry Pi OS, and an unconditional dependency there would break
`pip install -r requirements.txt` for the whole application on that
platform, not just narration. It lives in its own optional file instead:

```bash
uv pip install --python .venv/bin/python -r requirements-tts.txt
```

This installs the same `piper-tts>=1.7.0,<2.0` pinned there. The verification
below was run against `piper-tts==1.7.0`, which pulled in
`onnxruntime==1.29.0` and `pathvalidate==3.3.1`. `.venv/bin/piper` is a
console-script entry point (not on `PATH`; must be invoked as
`.venv/bin/piper`).

Without this package installed, the application still runs fully - every map
ships with its narration pre-generated - but you lose the ability to
(re)generate audio with `generate_audio` and the runtime fallback that
synthesizes on the fly for a map without pre-generated clips.

```bash
.venv/bin/python -c "import piper; print('piper module OK')"
which piper || ls .venv/bin/piper
```

Both succeeded: the module imports, and `.venv/bin/piper` exists (`which
piper` reports nothing because `.venv/bin` is not on `PATH`, which is
expected).

piper-tts 1.7.0 is GPL-3.0-or-later and publishes a `manylinux_2_17_aarch64`
wheel, so it installs the same way on 64-bit Raspberry Pi OS. **A 32-bit
Raspberry Pi OS is not covered by this wheel** and would need a standalone
piper binary instead (not verified here — no 32-bit target was available).

## Download the voice

```bash
mkdir -p models/tts_voices
BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main/uk/uk_UA/ukrainian_tts/medium
curl -L -o models/tts_voices/uk_UA-ukrainian_tts-medium.onnx      "$BASE/uk_UA-ukrainian_tts-medium.onnx"
curl -L -o models/tts_voices/uk_UA-ukrainian_tts-medium.onnx.json "$BASE/uk_UA-ukrainian_tts-medium.onnx.json"
```

Result: `uk_UA-ukrainian_tts-medium.onnx` is 76,735,663 bytes (76.7 MB, as
expected) and `uk_UA-ukrainian_tts-medium.onnx.json` is 2,002 bytes.

**Chosen voice: `uk_UA-ukrainian_tts-medium`.** It was not swapped for
`uk_UA-tetiana-high` or `uk_UA-lada-x_low` — see the WAV verification note
below for why that decision is still open.

`models/tts_voices/` is gitignored (see `.gitignore`), so the `.onnx` file
was **not** committed here — only this document.

## Synthesis sanity check (Step 3)

```bash
echo 'Хрещатик' | .venv/bin/piper \
  --model models/tts_voices/uk_UA-ukrainian_tts-medium.onnx \
  --output_file /tmp/probe.wav
```

This machine cannot listen to audio, so the WAV was checked structurally
instead of by ear:

- channels: 1
- sample rate: 22050 Hz
- frames: 18176
- duration: 0.824 s

That duration is plausible for a three-syllable word and far above silence,
so the file is real speech-shaped audio, not empty output. **A human still
needs to listen to `/tmp/probe.wav` (or a regenerated copy) to judge
intelligibility before `TTSConfig.VOICE` (`src/config.py`) locks in
`uk_UA-ukrainian_tts-medium` for production use.**

Contradiction found during this step: piper printed
`WARNING:piper.phoneme_ids:Missing phoneme from id map: Х` for the
capitalized first letter of `Хрещатик`. Re-running with the all-lowercase
`хрещатик` produced no such warning. So this voice's phoneme map appears to
be missing entries for capital Cyrillic letters — capitalized words (proper
nouns, sentence-initial words) will get a slightly wrong/degraded phoneme for
their first letter unless the input text is lowercased before synthesis.
This is handled in `src/tts/engine.py`'s `_phonemizable()`, which lowercases
every text before it reaches piper.

## Per-clip timing (Step 4)

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

Measured output, verbatim:

```
single: 1.59s total, 1.59s per clip
five separate processes: 9.94s total, 1.99s per clip
```

This machine is faster than a Raspberry Pi 4; the number that decides the
design is the one measured on the actual device, not this one. Note that each of the five
"separate processes" runs pays the full model-load cost again (no process
reuse), which is why the per-clip cost is close to (not far below) the
single-clip cost — one process per clip does not amortize load time across
clips.

## Batching probe (Step 5)

One proposed batching command, from the original design:

```bash
printf '%s\n' \
  '{"text": "Хрещатик", "output_file": "/tmp/batch_a.wav"}' \
  '{"text": "Сектор А", "output_file": "/tmp/batch_b.wav"}' \
| .venv/bin/piper --model models/tts_voices/uk_UA-ukrainian_tts-medium.onnx --json-input
```

**Result: `--json-input` is not a real flag in piper-tts 1.7.0.** `piper
--help` lists no `--json-input` option at all (full flag list: `-m/--model`,
`-c/--config`, `-i/--input-file`, `-f/--output-file`, `-d/--output-dir`,
`--output-dir-naming {timestamp,text}`, `--output-raw`, `-s/--speaker`,
`--length-scale`, `--noise-scale`, `--noise-w-scale`, `--cuda`,
`--sentence-silence`, `--volume`, `--no-normalize`, `--data-dir`, `--debug`).
Running the exact command above produced no `/tmp/batch_a.wav` or
`/tmp/batch_b.wav` — `ls` confirmed both were missing. So: **no, the
JSON-per-line stdin protocol shown above does not exist in this version.**

However, one process taking many clips is possible through a different,
real mechanism — `-i`/`--input-file` plus `-d`/`--output-dir` with
`--output-dir-naming text`:

```bash
printf 'Хрещатик\nСектор А\n' > /tmp/batch_input.txt
.venv/bin/piper --model models/tts_voices/uk_UA-ukrainian_tts-medium.onnx \
  -i /tmp/batch_input.txt -d /tmp/batchdir --output-dir-naming text
```

This produced both `/tmp/batchdir/Хрещатик.wav` (34,348 bytes) and
`/tmp/batchdir/Сектор А.wav` (30,252 bytes) from a single process
invocation. So batching multiple clips into one process **is** possible on
this piper version, just via `-i`/`-d`/`--output-dir-naming`, not
`--json-input`. `src/tts/engine.py` uses one process per clip regardless of
this result (an accepted degradation - see the design doc's "Out of scope");
this is only informative for whether a future change should add batching as
an optimization.

## Summary

- Voice for `TTSConfig.VOICE`: `uk_UA-ukrainian_tts-medium`
  (pending human listening confirmation of `/tmp/probe.wav`).
- Per-clip cost on this dev machine (a baseline only, not the Raspberry Pi
  number): ~1.59s cold single clip, ~1.99s/clip amortized over 5 separate
  processes.
- Batching via `--json-input`: not supported in piper-tts 1.7.0. A working
  alternative (`-i` + `-d --output-dir-naming text`) exists if batching is
  later worth adding.
- Uppercase Cyrillic phoneme gap: capitalized words trigger a "missing
  phoneme" warning for their first letter; lowercasing input avoids it.
- The aarch64 wheel covers 64-bit Raspberry Pi OS; a 32-bit OS would need a
  standalone piper binary instead (not tested here).
