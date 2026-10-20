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
from src.map_reader import resolve_model, planned_path

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
    model_path = Path(model_path).resolve()
    raw = model_path.read_text(encoding='utf-8')
    document = json.loads(raw)
    original_model = document['model']
    model = resolve_model(document, model_path)

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
        key = 'audioDescription' if entry.kind == 'hotspot' else 'map_description'
        old_path = (original_model['hotspots'][entry.index].get('audioDescription')
                    if entry.kind == 'hotspot' else original_model.get('map_description'))
        if not old_path:
            # The reader can infer an ID-owned destination from navigation.
            # Keep that recovery in memory instead of inserting JSON fields.
            continue
        stored_entry = model_audio.NarrationEntry(entry.kind, entry.index, entry.text, old_path)
        stored_output = model_audio.output_path(stored_entry)
        # Keep the maker's models/<name>/... strings when they already describe
        # the generated output. Only actual format/directory changes need edits.
        new_path = (str(stored_output) if planned_path(str(stored_output), model_path) == job.output_path
                    else job.output_path.relative_to(model_path.parent).as_posix())
        if new_path == old_path:
            continue
        text = json_edit.replace_string_value(text, key, old_path, new_path)
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
