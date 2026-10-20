"""Read the existing maker's loose-file exports without rewriting their JSON."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path, PurePosixPath
import re
import unicodedata
import wave

APP_ROOT = Path(__file__).resolve().parent.parent
SHARED_AUDIO = ('blipsound', 'welcome_message', 'goodbye_message', 'heartbeat', 'crickets')
AUDIO_SUFFIXES = {'.wav', '.mp3', '.ogg', '.flac', '.m4a'}


def _inside(base, path):
    path = path.resolve()
    if path != base and base not in path.parents:
        raise ValueError(f'Map asset escapes the map directory: {path}')
    return path


def local_paths(value, model_file):
    """Rebase models/<old-name>/... onto the JSON directory, never another map."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Map asset needs a nonempty filename')
    value = value.replace('\\', '/').strip()
    path = PurePosixPath(value)
    if '..' in path.parts or '://' in value:
        raise ValueError(f'Unsupported map asset path: {value}')
    base = Path(model_file).resolve().parent
    absolute = path.is_absolute() or bool(re.match(r'^[A-Za-z]:/', value))
    candidates = []
    if path.is_absolute() and (Path(value).resolve() == base or base in Path(value).resolve().parents):
        candidates.append(Path(value))
    indices = [i for i, part in enumerate(path.parts) if part == 'models']
    if indices and indices[-1] + 2 < len(path.parts):
        candidates.append(base.joinpath(*path.parts[indices[-1] + 2:]))
    if not absolute:
        candidates.append(base.joinpath(*path.parts))
    candidates.append(base / path.name)
    result = []
    for candidate in candidates:
        candidate = _inside(base, candidate)
        if candidate not in result:
            result.append(candidate)
    return result


def planned_path(value, model_file):
    """Destination for missing audio; used by the reader and TTS path writer."""
    return local_paths(value, model_file)[0]


def _label(value):
    return ''.join(c for c in unicodedata.normalize('NFC', value or '').casefold() if c.isalnum())


def _zone_id(hotspot, graph):
    identities = set()
    explicit = hotspot.get('id')
    if explicit is not None:
        if not isinstance(explicit, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', explicit):
            raise ValueError('Invalid zone ID')
        identities.add(explicit)
    stem = PurePosixPath((hotspot.get('audioDescription') or '').replace('\\', '/')).stem
    if re.fullmatch(r'zone-\d+', stem):
        identities.add(stem)
    if isinstance(graph, dict) and isinstance(graph.get('nodes'), list) and isinstance(hotspot.get('color'), list):
        for node in graph['nodes']:
            if isinstance(node, dict) and node.get('zone') == hotspot['color']:
                match = re.fullmatch(r'(zone-\d+)-node', str(node.get('id', '')))
                if match:
                    identities.add(match[1])
    if len(identities) > 1:
        raise ValueError(f'Conflicting zone identity in audio path / navigation: {sorted(identities)}')
    return next(iter(identities), None)


def _has_zone_ids(model):
    """Recognize ID-bearing exports without imposing a new JSON schema."""
    hotspots = model.get('hotspots', [])
    return bool(hotspots) and all('id' in hotspot for hotspot in hotspots)


def resolve_model(document, model_file, app_root=APP_ROOT):
    """Resolve existing maker/legacy paths, preserving future TTS destinations.

    No package version, canonical image filename or new JSON field is required.
    Missing image checks are separate so the TTS CLI can also handle narration-only
    models. Shared sounds are resolved independently from the application root.
    """
    if not isinstance(document, dict) or not isinstance(document.get('model'), dict):
        raise ValueError('Expected a JSON object containing a model object')
    model = copy.deepcopy(document['model'])
    source = Path(model_file).resolve()
    base = source.parent
    hotspots = model.get('hotspots', [])
    if not isinstance(hotspots, list) or not all(isinstance(h, dict) for h in hotspots):
        raise ValueError('hotspots must be an array of objects')
    for hotspot in hotspots:
        if not isinstance(hotspot.get('textDescription', ''), str):
            raise ValueError('Zone textDescription must be a string')
        if hotspot.get('audioDescription') is not None and not isinstance(hotspot['audioDescription'], str):
            raise ValueError('Zone audioDescription must be a string')
    for field in ('map_description', 'mapDescriptionText'):
        if model.get(field) is not None and not isinstance(model[field], str):
            raise ValueError(f'{field} must be a string')
    labels = Counter(_label(h.get('textDescription')) for h in hotspots)
    audio_files = None

    def audio(value, label=None, *, generated=None, exclude=()):
        nonlocal audio_files
        if value:
            for path in local_paths(value, source):
                if path.is_file():
                    return path
        if generated is not None and generated.is_file():
            return _inside(base, generated)
        if audio_files is None:
            audio_files = [p for p in base.rglob('*') if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES]
        stem = PurePosixPath(value.replace('\\', '/')).stem.casefold() if value else ''
        matches = [p for p in audio_files if stem and p.stem.casefold() == stem
                   and p.resolve() not in exclude]
        if not matches and label:
            matches = [p for p in audio_files if _label(p.stem) == _label(label)]
            if matches and labels[_label(label)] > 1:
                raise ValueError(f'Repeated zone label {label!r}; use separate zone-ID filenames')
        if len(matches) > 1:
            raise ValueError(f'Ambiguous audio for {value or label}: ' + ', '.join(str(p) for p in matches))
        return _inside(base, matches[0]) if matches else None

    for field in ('filename', 'template_image'):
        if field in model:
            candidates = local_paths(model[field], source)
            model[field] = str(next((p for p in candidates if p.is_file()), candidates[0]))
    for field in SHARED_AUDIO:
        value = model.get(field)
        if value:
            if not isinstance(value, str):
                raise ValueError(f'{field} must be a filename string')
            path = Path(value.replace('\\', '/')).expanduser()
            if not path.is_absolute() and '..' in path.parts:
                raise ValueError(f'Invalid shared audio path: {value}')
            model[field] = str((path if path.is_absolute() else Path(app_root) / path).resolve())

    identities = set()
    for hotspot in hotspots:
        zone_id = _zone_id(hotspot, model.get('navigation'))
        if zone_id:
            if zone_id.casefold() in identities:
                raise ValueError(f'Duplicate zone ID: {zone_id}')
            identities.add(zone_id.casefold())
            hotspot['id'] = zone_id
        value = hotspot.get('audioDescription') or ''
        effective = value or (f'Audio/tts/{zone_id}.wav' if zone_id else '')
        if effective:
            hotspot['audioDescription'] = str(planned_path(effective, source))
        else:
            raise ValueError('Cannot determine zone audio: supply its filename or a zone ID')

    if (_has_zone_ids(model) and 'mapDescriptionText' in model
            and not (model.get('mapDescriptionText') or '').strip()):
        model.pop('map_description', None)

    # Check original destinations before filename fallback can merge two missing
    # sources into one existing generated clip. Explicit shared recordings stay valid.
    from src.tts import model_audio

    if model.get('map_description'):
        model['map_description'] = str(planned_path(model['map_description'], source))
    entries = model_audio.narration_entries(model)
    conflicts = model_audio.conflicting_outputs(entries)
    if conflicts:
        raise ValueError('Conflicting narration outputs: ' + ', '.join(str(p) for p in sorted(conflicts)))

    for hotspot in hotspots:
        found = audio(hotspot['audioDescription'], hotspot.get('textDescription'))
        if found is not None:
            hotspot['audioDescription'] = str(found)

    if model.get('map_description'):
        value = model['map_description']
        description = next(entry for entry in entries if entry.kind == 'map_description')
        zone_outputs = {model_audio.output_path(entry) for entry in entries if entry.kind == 'hotspot'}
        found = audio(value, generated=model_audio.output_path(description), exclude=zone_outputs)
        if found is not None:
            model['map_description'] = str(found)
        elif (model.get('mapDescriptionText') or '').strip():
            model['map_description'] = str(planned_path(value, source))
        else:
            model.pop('map_description', None)
    else:
        model.pop('map_description', None)
    return model


def validate_images(model, strict=False):
    """Validate decoded sizes; strict preflight also checks exact RGB coverage."""
    import cv2 as cv
    import numpy as np

    for field in ('filename', 'template_image'):
        if not isinstance(model.get(field), str):
            raise ValueError(f'Missing {field}')
    mask = cv.imread(model['filename'], cv.IMREAD_COLOR)
    template = cv.imread(model['template_image'], cv.IMREAD_GRAYSCALE)
    for field, image in [('filename', mask), ('template_image', template)]:
        if image is None:
            raise ValueError(f'{field}: missing or unsupported image: {model[field]}; use PNG or JPEG')
    if mask.shape[:2] != template.shape[:2]:
        raise ValueError(f'Mask size {mask.shape[:2]} differs from template size {template.shape[:2]}')
    declared = set()
    for hotspot in model.get('hotspots', []):
        color = hotspot.get('color')
        if not isinstance(color, list) or len(color) != 3 or any(type(c) is not int or not 0 <= c <= 255 for c in color):
            raise ValueError('Hotspot color must be an RGB integer triple')
        packed = (color[0] << 16) | (color[1] << 8) | color[2]
        if packed in declared:
            raise ValueError('Duplicate hotspot color')
        declared.add(packed)
    if strict:
        present = set()
        for start in range(0, mask.shape[0], 128):
            stripe = mask[start:start + 128].astype(np.uint32)
            packed = (stripe[:, :, 2] << 16) | (stripe[:, :, 1] << 8) | stripe[:, :, 0]
            present.update(int(value) for value in np.unique(packed))
        if present - declared - {0xFFFFFF}:
            raise ValueError('Mask contains undeclared colors (or smoothed zone edges)')
        if declared - present:
            raise ValueError('Some declared zones have no pixels in the mask')


def read_map(filename, app_root=APP_ROOT):
    source = Path(filename).expanduser().resolve()
    document = json.loads(source.read_text(encoding='utf-8'))
    model = resolve_model(document, source, app_root=app_root)
    if model.get('modelType'):
        if model['modelType'] != 'sift_2d_mediapipe':
            raise ValueError(f"Unsupported modelType: {model['modelType']}")
        # ID-bearing maker maps have exact masks; older hand-painted maps can
        # have anti-aliased fringes and retain their previous behavior.
        if _has_zone_ids(model) and ('filename' in model or 'template_image' in model):
            validate_images(model, strict=True)
    return model


def main():
    parser = argparse.ArgumentParser(description='Check an existing maker export without starting the camera')
    parser.add_argument('--input1', required=True, help='Exported model JSON, not .camio-project.json')
    parser.add_argument('--allow-missing-audio', action='store_true', help='Check before generating narration')
    args = parser.parse_args()
    try:
        model = read_map(args.input1)
        validate_images(model, strict=True)
        problems = [f'Missing shared sound: {model[field]}' for field in SHARED_AUDIO
                    if model.get(field) and not Path(model[field]).is_file()]
        paths = [h['audioDescription'] for h in model.get('hotspots', [])]
        if model.get('map_description'):
            paths.append(model['map_description'])
        for path in paths:
            if not Path(path).is_file():
                if not args.allow_missing_audio:
                    problems.append(f'Missing narration: {path}')
            elif Path(path).suffix.lower() == '.wav':
                try:
                    with wave.open(path, 'rb') as clip:
                        size = clip.getnchannels() * clip.getsampwidth()
                        if clip.getnframes() == 0 or len(clip.readframes(1)) != size:
                            problems.append(f'Empty WAV: {path}')
                except (OSError, wave.Error, EOFError):
                    problems.append(f'Invalid WAV: {path}')
        for problem in problems:
            print(problem)
        if problems:
            return 1
        print(f"Map valid: {model.get('name') or Path(args.input1).stem}, {len(model.get('hotspots', []))} zones")
        return 0
    except (ValueError, OSError) as error:
        print(f'Map invalid: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
