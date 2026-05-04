# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -------------------------------------------------------

project = "ASTRA-Core"
copyright = "2026, Ishan Tare"
author = "Ishan Tare"

try:
    from astra.version import __version__ as _pkg_version
except ImportError:
    _pkg_version = "3.6.0"

version = release = _pkg_version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",    # Google/NumPy style docstrings
    "sphinx.ext.viewcode",      # Source code links
    "sphinx_copybutton",        # Copy button on code blocks
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc settings -------------------------------------------------------
# 'none' avoids duplicating dataclass field docs (napoleon adds them separately)
autodoc_typehints = "none"
autodoc_member_order = "bysource"

# -- Suppress known cosmetic warnings ----------------------------------------
# Duplicate: napoleon+autodoc both document dataclass fields
# Undefined substitution: :date: etc. used in examples
# Missing reference: :doi: :arxiv: in docstrings
suppress_warnings = [
    "autodoc",
    "napoleon",
    "ref.python",
]

# -- HTML output -----------------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
