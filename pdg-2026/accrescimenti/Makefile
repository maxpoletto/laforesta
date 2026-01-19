CSV_DIR=csv

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
# Data dependencies
#

DATA_DEPS=$(FILES_EQUAZIONI) $(CSV_DIR)/alberi.csv $(CSV_DIR)/alberi-calcolati.csv $(CSV_DIR)/particelle.csv $(CSV_DIR)/comparti.csv $(CSV_DIR)/provv_vol.csv $(CSV_DIR)/provv_eta.csv

#
# HTML output
#

FILES_HTML=html/index.html
DEPS_HTML=$(DATA_DEPS) acc.py template/html/style.css

html: $(FILES_HTML)

html/%.html: template/html/%.html $(DEPS_HTML)
	cp -f template/html/style.css html/
	./acc.py --report --formato html --particelle $(CSV_DIR)/particelle.csv --dati $(CSV_DIR) --input template/html/$*.html --output-dir html/

clean-html:
	rm -f $(FILES_HTML)

#
# LaTeX / PDF output
#

FILES_PDF=tex/relazione.pdf
FILES_TEX=tex/relazione.tex
DEPS_TEX=$(DATA_DEPS) acc.py

pdf: $(FILES_PDF)

tex/%.tex: template/tex/%.tex $(DEPS_TEX)
	./acc.py --report --formato tex --particelle $(CSV_DIR)/particelle.csv --dati $(CSV_DIR) --input template/tex/$*.tex --output-dir tex/

tex/%.pdf: template/tex/%.tex $(DEPS_TEX)
	./acc.py --report --formato pdf --particelle $(CSV_DIR)/particelle.csv --dati $(CSV_DIR) --input template/tex/$*.tex --output-dir tex/

clean-pdf:
	rm -f $(FILES_PDF) $(FILES_TEX)

#
# CSV output
#

FILES_CSV=csv/ripresa.csv
DEPS_CSV=$(DATA_DEPS) acc.py

csv: $(FILES_CSV)

$(CSV_DIR)/ripresa.csv: template/csv/ripresa.csv $(DEPS_CSV)
	./acc.py --report --formato csv --particelle $(CSV_DIR)/particelle.csv --dati $(CSV_DIR) --input template/csv/ripresa.csv --output-dir csv/

clean-csv:
	rm -f $(FILES_CSV)
