#!/usr/bin/env python3
"""Regenerate ``docs/apidoc/*.rst`` with sphinx-apidoc before ``sphinx-build``.

Read the Docs runs this from :file:`.readthedocs.yaml` ``post_install``.
Local builds: ``python docs/generate_api.py`` or ``make html`` / ``make.bat html``.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_DOCS = Path(__file__).resolve().parent
_ROOT = _DOCS.parent
_OUT = _DOCS / "apidoc"
_PKG = _ROOT / "astra"


def main() -> int:
    if not _PKG.is_dir():
        print(f"error: package path not found: {_PKG}", file=sys.stderr)
        return 1
    if _OUT.exists():
        shutil.rmtree(_OUT)
    _OUT.mkdir(parents=True)
    cmd = [
        sys.executable,
        "-m",
        "sphinx.ext.apidoc",
        "-o",
        str(_OUT),
        "-f",
        "-e",
        "--module-first",
        str(_PKG),
    ]
    print("generate_api:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_ROOT))
    _drop_root_automodule(_OUT / "astra.rst")
    return 0


def _drop_root_automodule(astra_rst: Path) -> None:
    """Remove ``automodule:: astra`` from the package index.

    The package root re-exports symbols; documenting it duplicates every
    class/function that submodule pages already document.
    """
    if not astra_rst.is_file():
        return
    text = astra_rst.read_text(encoding="utf-8")
    needle = "\n\nSubmodules\n"
    i = text.find(needle)
    if i == -1:
        return
    head = "astra package\n=============\n"
    astra_rst.write_text(head + text[i:], encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
