# Minimal makefile for Sphinx documentation
#
# HTML depends on apidoc refresh (same as Read the Docs post_install).

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help Makefile apidoc clean html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

apidoc:
	python generate_api.py

html: Makefile apidoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Other output formats, linkcheck, etc.:
#   python generate_api.py && sphinx-build -M <target> . _build
