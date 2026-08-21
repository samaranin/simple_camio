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

    audio_path may already be a previously generated path - the CLI rewrites
    audioDescription to point here, so every run after the first reads it back as
    the model's current audio path. Appending another subdir in that case would
    nest tts/tts/... deeper on every run, so an audio_path already inside subdir
    is returned as-is (with a .wav suffix) instead.
    """
    path = Path(audio_path)
    subdir = subdir or TTSConfig.GENERATED_SUBDIR
    if path.parent.name == subdir:
        return path.with_suffix('.wav')
    return path.parent / subdir / (path.stem + '.wav')


#: Fixed output stem for a model's description, which has no basename of its own.
MAP_DESCRIPTION_STEM = 'map_description'


def output_path(entry, subdir=None):
    """
    Where this entry's generated WAV belongs.

    A hotspot keeps its source file's basename, so each generated clip stays
    traceable to the recording it replaces. The map description gets a fixed name
    instead: two of the three real models point map_description at the same file as
    one of their hotspots, and deriving from that basename would send two different
    texts to one output file.
    """
    if entry.kind == 'hotspot':
        return generated_path(entry.audio_path, subdir)
    parent = Path(entry.audio_path).parent
    subdir = subdir or TTSConfig.GENERATED_SUBDIR
    if parent.name == subdir:
        return parent / f'{MAP_DESCRIPTION_STEM}.wav'
    return parent / subdir / f'{MAP_DESCRIPTION_STEM}.wav'


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
        if force or not output_path(entry, subdir).is_file():
            result.append(entry)
    return result
