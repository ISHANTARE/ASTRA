# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ASTRA-Core'
copyright = '2026, Ishan Tare'
author = 'Ishan Tare'

try:
    from astra.version import __version__ as _pkg_version
except ImportError:
    _pkg_version = '3.3.0'

version = release = _pkg_version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',    # Parses Google and NumPy style docstrings
    'sphinx.ext.viewcode',    # Adds links to highlighted source code
    'sphinx_copybutton',      # Adds a literal copy button to code blocks
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Avoid duplicate attribute lines for dataclasses / type-hinted fields (napoleon + autodoc).
autodoc_typehints = 'none'

suppress_warnings = [
    'ref.python',  # :class:`AstraError` etc. exposed both on astra and astra.errors
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
