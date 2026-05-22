# Makefile for compiling LaTeX CVs

# Find all .tex files in the current directory
TEX_FILES = $(wildcard *.tex)

# Define PDF targets based on the .tex files
PDF_FILES = $(TEX_FILES:.tex=.pdf)

# Default target: compile all found latex files
all: $(PDF_FILES)

# Rule to compile a generic .pdf from a .tex file
%.pdf: %.tex
	pdflatex $<

# Clean up build artifacts
clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc *.vrb *.snm *.nav *.fls *.fdb_latexmk *.texbak

# Clean everything including generated PDFs
clean-all: clean
	rm -f *.pdf

# Phony targets to ensure these aren't treated as files
.PHONY: all clean clean-all