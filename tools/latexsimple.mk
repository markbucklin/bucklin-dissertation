PDFTEX		= pdflatex
HTMLTEX		= htlatex
HTMLTEX_FLAGS	= 'html, NoFonts'
CSSFILE		= css/custom.css
TEXFILES	= $(wildcard *.tex)
CONTENT		= $(wildcard section/*.tex) $(wildcard preamble/*.tex) 
AUXFILES	= $(wildcard preamble/*.aux section/*.aux *.4ct *.4tc *.aux *.bbl *.bcf *.blg *.dvi *.idv *.lg *.log *.out *.png *.run.xml *.svg *.tmp *.xref)
PDF		= $(TEXFILES:.tex=.pdf)
HTML		= $(TEXFILES:.tex=.html)

.PHONY		= clean wipe html

html: $(TEXFILES) $(HTML)

%.html: %.tex $(CSSFILE) $(CONTENT) 
	$(HTMLTEX) $< $(HTMLTEX_FLAGS) && cat $(CSSFILE) >> $(<:.tex=.css)	



pdf: $(TEXFILES) $(PDF) $(CONTENT)

%.pdf: %.tex $(CONTENT)
	$(PDFTEX) $<
	biber $(<:.tex=.bcf) 
	$(PDFTEX) $<



clean:
	rm -f $(AUXFILES)

wipe: clean
	rm -f $(PDF) $(HTML) $(HTML:.html=.css) 
