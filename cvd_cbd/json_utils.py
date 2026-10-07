"""Standard JSON decoding with object-local duplicate-key rejection."""
import json
from pathlib import Path


class DuplicateJSONKeyError(ValueError):
    """Two decoded member names occur in the same JSON object."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            # Keys as well as values can contain sensitive user data.
            raise DuplicateJSONKeyError('Duplicate JSON key in an object')
        result[key] = value
    return result


def loads_json(text, *, source='JSON input'):
    """Preserve json.loads types and values; reject every duplicate object member.

    The standard decoder resolves escapes before calling object_pairs_hook.
    A fresh dictionary per object allows repeated names in separate objects.
    Diagnostics identify the input, without echoing keys or document contents.
    """
    try:
        return json.loads(text, object_pairs_hook=_unique_object)
    except DuplicateJSONKeyError as exc:
        raise DuplicateJSONKeyError(f'{source}: {exc}') from None
    except json.JSONDecodeError as exc:
        raise ValueError(f'{source}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}') from None


def read_json(path, *, encoding=None):
    """Read once using the caller's existing text encoding policy."""
    return loads_json(Path(path).read_text(encoding=encoding), source=str(path))
