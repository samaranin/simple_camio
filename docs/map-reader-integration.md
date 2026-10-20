# Maker map reader and TTS integration

The reader accepts the existing maker's loose-file exports. It resolves
`models/<name>/...` paths relative to the exported JSON, identifies zones from
existing filenames/navigation, and passes absolute paths to CamIO in memory.
The maker and map JSON schema are unchanged.

## Compatibility

- Standard maker exports retain byte-identical JSON during narration generation
  and folder relocation. Empty maker descriptions disable their placeholder.
- Legacy models retain their loading behavior, including different template and
  mask dimensions; an explicit reader preflight still validates images strictly.
- Existing legacy description recordings remain available without description text.
- Both runtime and offline TTS reuse usable recordings. `--force` explicitly
  regenerates narration through the offline CLI.
- Runtime updates playback paths in memory after successful synthesis and reuses
  generated clips on later launches, including when runtime synthesis is disabled.
- Legacy overviews use their distinct generated `map_description.wav`, rather than
  accidentally adopting a hotspot recording with a shared original filename.
- Ambiguous generated destinations are rejected by the reader. Runtime skips
  colliding replacements without overwriting or reusing another zone's recording;
  independent clips can still be generated.

No map migration is required. Existing model files, images and recordings were
not changed. New local maps, generated audio, voice models, test-copy folders,
Python environments and unrelated macOS configuration changes are excluded.

## Validation

A clean source snapshot without local macOS changes, new maps, or a sibling maker
checkout passed **203 tests, with one optional test skipped**, on 2026-10-20.
Core reader tests create temporary input files and require neither Node.js nor
the maker. Only the optional comparison with the actual maker exporter skips
when that checkout or Node.js is unavailable; that comparison also passed in
this workspace.

Regression coverage includes legacy CnapMap loading, strict maker validation,
unchanged JSON after relocation and synthesis, original recording reuse,
forced generation, partial synthesis failures, repeated runtime loading,
legacy descriptions, and colliding/explicitly owned narration destinations.
Synthesis tests use temporary files; the existing maps were only read. SHA-256
checks confirmed that all 173 existing files under models/ remained unchanged.

Pytest returned exit code 0. Pyglet still prints a display-related interpreter
shutdown traceback in the sandbox without a Cocoa display; physical camera
tracking, audible playback and Raspberry Pi operation require device testing.

## Branch and future PR

The reader branch is based on upstream `auto-narration` commit
`d69e05e89bb0802dfbf82771614d7be1107acc4e`. Publish it to
`tByte97/simple_camio:reader-tts-integration`; do not create a PR yet.

A later reader-only PR can target `samaranin/simple_camio:auto-narration`, or
`master` after upstream TTS PR #8 is merged. A PR to today's master would also
include the upstream TTS commits.
