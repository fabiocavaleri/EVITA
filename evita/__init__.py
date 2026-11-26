"""
EVITA. Its repository root is put on Python's module search path: EVITA imports DoppelTest's configuration as
EVITA_DoppelTest.config, a folder of the repository rather than an installed package. Editable installs made by
recent pip / setuptools expose only the `evita` package (not the repository root), so without this the import works
or fails depending on the working folder and the install mode (e.g. `evita run`, or EVITA's search started from evita/).
"""
import os as _os
import sys as _sys

_REPOSITORY_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPOSITORY_ROOT not in _sys.path:
    _sys.path.append(_REPOSITORY_ROOT)