"""Tokenizing and light normalization shared by retrieval and evaluation."""
from __future__ import annotations

import re

_PHRASES = [
    (r"mach[\s-]*zehnder(\s+interferometer)?", "mzi"),
    (r"multi[\s-]*mode(\s+interferometer)?", "mmi"),
    (r"thermo[\s-]*optic", "thermooptic"),
    (r"ring\s+resonator", "ring"),
    (r"directional\s+coupler", "coupler"),
    (r"photo[\s-]*(detector|diode)", "photodetector"),
]
_STOP = {
    "a", "an", "the", "of", "with", "and", "to", "for", "in", "on", "at", "by", "is", "are",
    "as", "each", "its", "that", "this", "from", "into", "use", "using", "per", "up",
}


def normalize(text: str) -> str:
    t = text.lower().replace("_", " ").replace("×", "x")
    t = re.sub(r"(\d+)\s*x\s*(\d+)", r"\1x\2", t)  # "1 x 2" -> "1x2"
    for pat, repl in _PHRASES:
        t = re.sub(pat, repl, t)
    return t


def tokenize(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", normalize(text)):
        if w in _STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


# order matters: more specific kinds first
_KINDS = [
    ("edge_coupler", r"edge\s*coupler"),
    ("grating_coupler", r"grating"),
    ("mzi", r"\bmzi\b"),
    ("mmi", r"\bmmi\b"),
    ("ring", r"\bring\b"),
    ("y_branch", r"\by\s*(branch|junction|splitter)\b"),
    ("photodetector", r"photodetector|\bdetector\b"),
    ("heater", r"heater|thermo"),
    ("coupler", r"coupler"),
    ("waveguide", r"waveguide|\bstraight\b"),
    ("bend", r"\bbend"),
]


def canonical_kind(text: str) -> str:
    """Map free-form component text ('MZI with thermo-optic heater') to a short canonical kind."""
    t = normalize(text)
    for name, pat in _KINDS:
        if re.search(pat, t):
            return name
    toks = tokenize(text)
    return toks[0] if toks else ""
