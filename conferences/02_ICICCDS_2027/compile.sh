#!/usr/bin/env bash
set -e
echo "Compiling ICIPCN 2027 IEEE Conference Paper..."
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdfinfo main.pdf | grep "Pages:"
echo "✅ Compilation successful! main.pdf is ready."
