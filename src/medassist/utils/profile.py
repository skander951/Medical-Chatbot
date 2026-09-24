"""Patient-profile helpers (pure Python, no LLM)."""
from __future__ import annotations

FIELD_LABELS = {
    "age": "age",
    "sex": "sex",
    "symptoms": "main symptoms",
    "allergies": "known allergies (or 'none')",
    "current_medications": "current medications (or 'none')",
}

_EMPTY = {"", "null", "unknown", "n/a", "not mentioned", "not specified"}


def _clean(value):
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return None if v.lower() in _EMPTY else v
    return value


def merge_profile(existing: dict, update: dict) -> dict:
    """Merge newly extracted fields into the profile without erasing known ones."""
    merged = dict(existing or {})
    for key in FIELD_LABELS:
        new = _clean((update or {}).get(key))
        if new is None:
            continue
        old = merged.get(key)
        if key == "symptoms" and old and str(new).lower() not in str(old).lower():
            merged[key] = f"{old}; {new}"  # accumulate symptoms across turns
        else:
            merged[key] = new
    return merged


def missing_fields(profile: dict) -> list[str]:
    profile = profile or {}
    return [k for k in FIELD_LABELS if profile.get(k) in (None, "")]
