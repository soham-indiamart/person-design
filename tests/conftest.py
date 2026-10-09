import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from vani_persona.config import load_thresholds  # noqa: E402
from vani_persona.orchestration.dialogue_state import ObjectivePolicy  # noqa: E402
from vani_persona.personas import PersonaRegistry  # noqa: E402


@pytest.fixture(scope="session")
def thresholds():
    return load_thresholds()


@pytest.fixture(scope="session")
def registry():
    return PersonaRegistry()


@pytest.fixture(scope="session")
def policy():
    return ObjectivePolicy.load("fix_meeting")
