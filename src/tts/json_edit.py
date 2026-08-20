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

    # Ambiguity is judged on the value itself: if the same value string shows up
    # more than once anywhere in the file -- even under a different key -- a
    # blind text.replace() could land on the wrong occurrence, so refuse rather
    # than guess.
    occurrences = text.count(old_literal)
    if occurrences > 1:
        raise JsonEditError(
            f'{key}={old_value!r} appears {occurrences} times; '
            f'cannot edit it unambiguously'
        )

    # The models use both "key":"value" and "key": "value".
    for gap in ('', ' '):
        needle = f'{key_literal}:{gap}{old_literal}'
        if needle in text:
            return text.replace(needle, f'{key_literal}:{gap}{new_literal}', 1)

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
