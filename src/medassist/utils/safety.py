"""Deterministic safety layer: emergency detection + fixed messages (no LLM involved)."""
from __future__ import annotations

import re

_EMERGENCY_PATTERNS = [
    r"chest (pain|pressure|tightness)",
    r"(can'?t|cannot|unable to) breathe",
    r"difficulty breathing|trouble breathing",
    r"severe bleeding|bleeding (heavily|a lot)|won'?t stop bleeding",
    r"unconscious|passed out|not responding",
    r"stroke|face (is )?drooping|slurred speech",
    r"seizure|convulsion",
    r"suicid|kill myself|end my life|want to die",
    r"overdose|took too many (pills|tablets)",
    r"anaphyla|throat (is )?(closing|swelling)",
]
_EMERGENCY_RE = re.compile("|".join(_EMERGENCY_PATTERNS), re.IGNORECASE)


def is_emergency(text: str) -> bool:
    return bool(text and _EMERGENCY_RE.search(text))


EMERGENCY_MESSAGE = (
    "**This may be a medical emergency.** Please call your local emergency number "
    "(for example 112 / 911 / 15) or go to the nearest emergency department right now. "
    "If you are thinking about harming yourself, contact a crisis line or emergency services immediately. "
    "I can't safely help with this through chat."
)

DISCLAIMER = (
    "_This assistant is for educational purposes only and is not a substitute for "
    "professional medical advice, diagnosis or treatment. Consult a qualified healthcare professional._"
)
