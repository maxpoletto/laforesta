CSV_DIR=csv

include Legacy.mak

.PHONY: \
	equazioni clean-equazioni \
	altvol clean-altvol \
	html clean-html \
	pdf clean-pdf \
    all clean

all: equazioni altvol pdf

clean: clean-equazioni clean-altvol clean-pdf

#
# Fit curves
#

FILES_EQUAZIONI=$(CSV_DIR)/equazioni_originali.csv $(CSV_DIR)/equazioni_ipsometro.csv $(CSV_DIR)/equazioni_tabelle.csv

equazioni: $(FILES_EQUAZIONI)

$(CSV_DIR)/equazioni_originali.csv: $(CSV_DIR)/alberi.csv
	./acc.py --genera-equazioni --funzione log --fonte-altezze originali --input $(CSV_DIR)/alberi.csv --output $(CSV_DIR)/equazioni_originali.csv

$(CSV_DIR)/equazioni_ipsometro.csv: $(CSV_DIR)/altezze.csv
	./acc.py --genera-equazioni --funzione log --fonte-altezze ipsometro --input $(CSV_DIR)/altezze.csv --output $(CSV_DIR)/equazioni_ipsometro.csv

$(CSV_DIR)/equazioni_tabelle.csv: $(CSV_DIR)/alsometrie.csv $(CSV_DIR)/particelle.csv
	./acc.py --genera-equazioni --funzione log --fonte-altezze tabelle --particelle $(CSV_DIR)/particelle.csv --input $(CSV_DIR)/alsometrie.csv --output $(CSV_DIR)/equazioni_tabelle.csv

clean-equazioni:
	rm -f $(FILES_EQUAZIONI)

#
# Heights and volumes
#


altvol: $(CSV_DIR)/alberi-calcolati.csv

$(CSV_DIR)/alberi-calcolati.csv: $(CSV_DIR)/alberi.csv $(CSV_DIR)/equazioni_ipsometro.csv
	./acc.py --calcola-altezze-volumi --equazioni $(CSV_DIR)/equazioni_ipsometro.csv --input $(CSV_DIR)/alberi.csv --output $(CSV_DIR)/alberi-calcolati.csv

clean-altvol:
	rm -f $(CSV_DIR)/alberi-calcolati.csv

#
# HTML output
#

FILES_HTML=html/per-compresa/classi-diametriche/index.html html/per-compresa/altezze-originali/index.html html/per-compresa/altezze-calcolate/index.html

html: $(FILES_HTML)

html/per-compresa/classi-diametriche/index.html: $(CSV_DIR)/alberi.csv $(CSV_DIR)/particelle.csv $(CSV_DIR)/equazioni_ipsometro.csv template/html/per-compresa/classi-diametriche/index.html html/style.css
	./acc.py --report --formato html --equazioni $(CSV_DIR)/equazioni_ipsometro.csv --alberi $(CSV_DIR)/alberi.csv --particelle $(CSV_DIR)/particelle.csv --input template/html/per-compresa/classi-diametriche/index.html --output-dir html/per-compresa/classi-diametriche/

html/per-compresa/altezze-originali/index.html: $(CSV_DIR)/alberi.csv $(CSV_DIR)/particelle.csv $(CSV_DIR)/equazioni_originali.csv template/html/per-compresa/altezze-originali/index.html html/style.css
	./acc.py --report --formato html --equazioni $(CSV_DIR)/equazioni_ipsometro.csv --alberi $(CSV_DIR)/alberi.csv --particelle $(CSV_DIR)/particelle.csv --input template/html/per-compresa/altezze-originali/index.html --output-dir html/per-compresa/altezze-originali/

html/per-compresa/altezze-calcolate/index.html: $(CSV_DIR)/alberi-calcolati.csv $(CSV_DIR)/particelle.csv $(CSV_DIR)/equazioni_ipsometro.csv template/html/per-compresa/altezze-calcolate/index.html html/style.css
	./acc.py --report --formato html --equazioni $(CSV_DIR)/equazioni_ipsometro.csv --alberi $(CSV_DIR)/alberi-calcolati.csv --particelle $(CSV_DIR)/particelle.csv --input template/html/per-compresa/altezze-calcolate/index.html --output-dir html/per-compresa/altezze-calcolate/

clean-html:
	rm -f $(FILES_HTML)

#
# LaTeX / PDF output
#

pdf: tex/relazione.pdf

DEPS_TEX=$(FILES_EQUAZIONI) $(CSV_DIR)/alberi.csv $(CSV_DIR)/alberi-calcolati.csv $(CSV_DIR)/particelle.csv template/tex/relazione.tex

tex/relazione.tex: $(DEPS_TEX)
	./acc.py --report --formato tex --particelle $(CSV_DIR)/particelle.csv --dati $(CSV_DIR) --input template/tex/relazione.tex --output-dir tex/

tex/relazione.pdf: $(DEPS_TEX)
	./acc.py --report --formato pdf --particelle $(CSV_DIR)/particelle.csv --dati $(CSV_DIR) --input template/tex/relazione.tex --output-dir tex/

clean-pdf:
	rm -f tex/relazione.pdf tex/relazione.tex
