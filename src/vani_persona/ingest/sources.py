"""Source dispatch: every CLI accepts ``--source`` so new export formats plug in without touching the pipeline."""

from __future__ import annotations

from pathlib import Path

from vani_persona.canonical import CallRecord

SOURCES = ("generic", "ps07")


def load_calls_from(path: Path, source: str = "generic") -> list[CallRecord]:
    if source == "generic":
        from vani_persona.ingest.adapter import load_calls

        return load_calls(path)
    if source == "ps07":
        from vani_persona.ingest.ps07 import load_ps07

        return load_ps07(path)
    raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
