"""Smoke test of the avcore package installation"""

from pathlib import Path

import avcore


def test_avcore_is_imported_from_src() -> None:
    """Smoke test of the avcore package installation"""

    package_file = Path(avcore.__file__)

    assert package_file.parts[-3:] == ("src", "avcore", "__init__.py")
