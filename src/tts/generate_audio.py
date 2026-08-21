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
        engine.SynthesisJob(text=entry.text, output_path=model_audio.output_path(entry))
        for entry in todo
    ]

    # A collision here would silently give a zone the wrong narration, so refuse
    # rather than let the last writer win.
    outputs = [job.output_path for job in jobs]
    if len(outputs) != len(set(outputs)):
        duplicates = {p for p in outputs if outputs.count(p) > 1}
        for path in sorted(duplicates):
            logger.error(f'two narration entries both target {path}')
        return len(jobs)

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
