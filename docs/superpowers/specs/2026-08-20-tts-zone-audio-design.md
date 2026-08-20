# TTS-generated zone audio

**Date:** 2026-08-20
**Status:** approved, ready for implementation planning
**Branch:** `auto-narration`

## Problem

Every interactive zone on a map needs a spoken description, and today each one is a
hand-recorded MP3 committed to the repository: 18 for UkraineMap, 27 for CnapMap,
7 for Heart. Authoring a new map means recording a clip per zone, so the cost of
adding or renaming a zone is a studio session rather than a text edit.

The zones already carry the text. Every hotspot in all three models has a
`textDescription`, used today only for log lines and the on-screen zone name.
Speech can be synthesized from it.

## Goal

Generate zone narration with text-to-speech instead of recording it, in Ukrainian,
on hardware that includes a Raspberry Pi 4.

## Decisions

| Question | Decision | Why |
| --- | --- | --- |
| Where does synthesis run? | Ahead of time, off-device; live synthesis on the Pi only as a fallback when a file is missing | Playback latency stays zero on the happy path, and voice quality is not capped by Pi CPU. The fallback keeps a freshly-copied map usable without a generation step. |
| Language | Ukrainian | Matches the maps' audience. |
| Text source | Rewrite the existing `textDescription` fields to Cyrillic | Single source of truth; no parallel field to drift. The existing values are Latin transliteration, unusable as TTS input either way. |
| Engine | Piper, both paths | Built for Raspberry Pi, ships an aarch64 wheel, has five `uk_UA` voices. One engine means the fallback sounds identical to the pre-generated files. |
| Fallback timing | At model load, synthesize everything missing at once | A stall while the user's finger rests on a zone is indistinguishable from a broken device; a one-off longer startup is not. |
| Scope | Hotspot zones plus the map description | Both are narration driven by text. Welcome and goodbye stay recorded — they are shared across maps and have no text field. |

## Architecture

Two paths over one engine:

```
AHEAD OF TIME (workstation, network available)
  map.json --> generate_audio CLI --> src/tts/engine --> piper --> <audio dir>/tts/*.wav
                     |
                     +--> rewrites audioDescription for the entries it generated

ON THE PI (no network, files already present)
  ZoneAudioPlayer --> file present? --> play (unchanged, zero latency)
                        |
                        +-- absent --> src/tts/engine --> piper --> cache at that path
                                          (at load, sequential, once)
```

Piper runs as a **subprocess**, not through its Python API. This keeps the ONNX
runtime out of the main process alongside MediaPipe, keeps a GPL-3.0 dependency at
arm's length, and leaves the engine replaceable without touching the player.

Startup fallback must hand every pending text to a **single** piper process. One
process per zone reloads the 60-100 MB voice model each time. Whether the 1.7.0 CLI
accepts a batch in the shape assumed here is unverified; if it does not, the
fallback degrades to one process per zone, which is slower but correct.

## Data model

Changes to each model JSON:

| Field | Before | After |
| --- | --- | --- |
| `textDescription` (x52) | `"Live peredserdya"` | `"Ліве передсердя"` |
| `audioDescription` (x52) | `models/Heart/Sound/Aorta.mp3` | `models/Heart/Sound/tts/Aorta.wav` |
| `mapDescriptionText` | absent | new: the text behind `map_description` |

- **Generated audio goes to a `tts/` subdirectory of the directory the existing
  `audioDescription` points into**, so `Audio/tts/` for UkraineMap and CnapMap but
  `Sound/tts/` for Heart, which names its audio directory differently. Recorded MP3s
  are left in place. Reverting
  to a recording, or comparing the two, is a one-field JSON edit rather than git
  archaeology. Output filenames reuse the existing basename, so each generated file
  is traceable to the recording it replaces.
- **WAV, not MP3.** Piper emits WAV; writing WAV bytes under an `.mp3` name is a lie
  someone trips over later. Both audio backends load WAV — `MP3/quick_blip.wav`
  already proves it — and it avoids putting ffmpeg on the Pi.
- **The voice model is not committed.** It lives in a gitignored directory, fetched by
  a documented command. Its path and the voice name live in a new `TTSConfig` in
  `src/config.py`, beside the other config classes.

## Components

| Module | Responsibility | Depends on |
| --- | --- | --- |
| `src/tts/engine.py` | Texts to WAV files. The only place that knows Piper exists: binary discovery, voice path, batch invocation | piper binary and voice files only. Knows nothing of models, hotspots, or JSON |
| `src/tts/generate_audio.py` | CLI: walk a model, compute what is missing, delegate to the engine, rewrite the JSON | `engine`, `json`. Knows nothing of audio backends |
| `TTSConfig` in `src/config.py` | `VOICE`, `VOICES_DIR`, `PIPER_BIN`, `RUNTIME_FALLBACK` (default on), `GENERATED_SUBDIR` (default `tts`) | — |
| `src/audio/audio.py` | New `_load_sound(path)` plus the fallback hook | `engine`, imported lazily |

The lazy import is load-bearing: with Piper absent the application must still start
and play whatever files exist.

The two paths differ in what they are allowed to write. The CLI writes audio **and**
edits the JSON, rewriting `audioDescription` only for the entries it actually
generated. The runtime fallback writes audio only, to the path the JSON already
names; it never edits a model file. A device in the field does not rewrite its own
configuration.

**One targeted cleanup is part of this work.** Sound loading in `audio.py` is
duplicated across `if USE_PYGLET / elif USE_PYGAME` branches in eight places. The
fallback is needed for both zones and the map description, so without a shared
helper its logic would be duplicated twice more. Extracting `_load_sound(path)`
gives all five consumers — zones, map description, welcome, goodbye, blip — one
code path, and one place for the fallback to live.

That also fixes an existing bug: `map_description` is loaded with no existence check
(`audio.py:200`), unlike the hotspots. On a fresh Pi where the generated description
is not yet present, the application would raise during startup. Through the helper
it simply starts without a description.

## Error handling

This is an assistive device: falling silent is cheaper than crashing. No TTS failure
may take the application down.

- **Piper or the voice model missing on the Pi** — one clear WARNING naming what is
  absent and the command that fixes it. Zones without files stay silent, everything
  else plays, the main loop starts.
- **Synthesis fails for one zone** — only that zone is silent, and the log says which.
  Other zones are unaffected.
- **Long startup** — progress goes to the log (`synthesizing 12/27`), otherwise an
  operator watching the journal sees a hang.
- **Empty or missing `textDescription`** — skipped with a warning rather than
  synthesizing silence.
- **Partial generation in the CLI** — non-zero exit and a list of the zones that
  failed, so it cannot be mistaken for success.
- **No silent overwrites.** By default the CLI fills gaps only and never replaces
  existing audio; replacing requires an explicit `--force`.

## Testing

The project has no pytest, no CI and no test config, so this introduces the
infrastructure: `pytest` as a dev dependency, and tests for the new module only.
`src/tap_classifier/test_tap_classifier.py` is left alone — rewriting it is separate
work and does not belong in this change.

Covered here, with no Pi and no voice model:

- **Planner** — given a model, which entries need synthesis and to which paths. Pure
  function.
- **Path derivation** — `Audio/X.mp3` to `Audio/tts/X.wav`, and `Sound/X.mp3` to
  `Sound/tts/X.wav`, so the Heart model's differently-named audio directory is covered.
- **JSON rewriting** — `audioDescription` updated, the rest of the file byte-identical,
  key order preserved.
- **Engine against a stub piper** — a fake binary that writes a tiny valid WAV, so
  batch invocation, per-item failure isolation, and the "piper unavailable" path are
  all tested without the voice model.
- **The no-overwrite invariant** — an existing file is untouched without `--force`.
  Locked down by its own test; a silent overwrite has already cost this project once.
- **Player fallback** — with a stub engine: a missing file triggers synthesis exactly
  once, and an engine that raises leaves the application usable.

## Verification on the device

These cannot be settled on a workstation and are steps, not assumptions:

1. `uname -m` — OS bitness, which decides whether Piper installs as a prebuilt wheel.
   4 GB of RAM does not imply a 64-bit OS; Pi 4 shipped 32-bit by default for years.
2. Synthesis wall-time per zone on the Pi 4. This decides whether the load-time
   fallback is tolerable at all: at 3 s per zone, 27 zones means 80 s of silence at
   startup, which is not a fallback and would send us back to background generation.
3. Whether the Piper 1.7.0 CLI accepts the assumed batch invocation.
4. Ukrainian pronunciation quality of the chosen voice — judged by a person, not a test.
5. Behaviour under systemd with the real audio device.

## Content migration

52 `textDescription` values plus 3 new `mapDescriptionText`, in three tiers of
difficulty:

- **Heart (7)** and **UkraineMap (18)** — mechanical de-transliteration.
  "Live peredserdya" to "Ліве передсердя", "Khreschatyk St." to "Хрещатик".
- **CnapMap (27)** — harder. Some values are genuinely English and need translating
  ("Passport Services" to "Паспортні послуги"); others read like labels from physical
  signage ("Sector A", "Stairs to the top"), and only the map's owner knows what the
  sign actually says.

This lands as a commit separate from the code, so the JSON `git diff` is the review
surface: old line above new, nothing else to read.

The risk here is not technical. The maps have printed physical counterparts, and the
narration has to match what is printed and what is on the signage. The draft is
reviewed by the project owner, and the migration is not complete until they confirm it.

## Out of scope

- Welcome and goodbye narration: shared across maps, no text field, still recorded.
- The `blipsound` effect: not speech.
- Removing the existing recorded MP3s: they stay on disk as a fallback and for
  comparison.
- Rewriting the existing tap-classifier test script.
- Cloud TTS: the systemd unit is deliberately free of network dependencies, and the
  on-device fallback has to work offline.
