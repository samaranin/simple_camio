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
