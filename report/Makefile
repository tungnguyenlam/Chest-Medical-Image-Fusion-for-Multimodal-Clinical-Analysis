# Makefile for compiling the LaTeX report

# Output PDF target
TARGET = main.pdf

# Root source file
ROOT_TEX = main.tex

# Find all .tex files recursively in the project to track dependencies
TEX_FILES = $(shell find . -name "*.tex")

# Find all bibliography files
BIB_FILES = $(wildcard *.bib)

# Find all images (if img directory exists)
IMG_FILES = $(shell find img -type f 2>/dev/null)

# Default target
all: $(TARGET)

# Rule to build the PDF target
$(TARGET): $(TEX_FILES) $(BIB_FILES) $(IMG_FILES)
	@echo "=================================================="
	@echo "Compiling LaTeX document (Pass 1)..."
	@echo "=================================================="
	pdflatex -interaction=nonstopmode $(ROOT_TEX)
	@if [ -f main.aux ] && grep -q "citation" main.aux; then \
		echo "=================================================="; \
		echo "Compiling Bibliography (BibTeX)..."; \
		echo "=================================================="; \
		bibtex main.aux; \
		echo "=================================================="; \
		echo "Compiling LaTeX document (Pass 2)..."; \
		echo "=================================================="; \
		pdflatex -interaction=nonstopmode $(ROOT_TEX); \
	fi
	@echo "=================================================="
	@echo "Compiling LaTeX document (Pass 3/Final)..."
	@echo "=================================================="
	pdflatex -interaction=nonstopmode $(ROOT_TEX)

# Clean up build artifacts (retaining the PDF)
clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc *.vrb *.snm *.nav *.fls *.fdb_latexmk *.texbak
	find . -name "*.aux" -delete

# Clean everything including generated PDFs
clean-all: clean
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean clean-all