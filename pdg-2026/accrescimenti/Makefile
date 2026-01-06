CSV_DIR=csv
OTHER_FLAGS=--un-genere-per-grafico

.PHONY: all equazioni \
	html clean-html \
	cd-per-compresa cd-per-particella \
	ci-originali-per-compresa ci-originali-per-particella \
	ci-interpolate-per-compresa ci-interpolate-per-particella \
	dati sync ipsometrie

all: dati

alsometrie-calcolate.csv: analysis.py $(CSV_DIR)/alsometrie.csv
	ls $(CSV_DIR)/alsometrie.csv
	./analysis.py --input-dir $(CSV_DIR) --genera-alsometrie-calcolate --fonte-altezze alsometrie --file-alsometrie-calcolate $(CSV_DIR)/alsometrie-calcolate.csv

alberi-calcolati.csv: analysis.py $(CSV_DIR)/alberi.csv $(CSV_DIR)/alsometrie.csv
	./analysis.py --input-dir $(CSV_DIR) --genera-alberi-altezze-calcolate --fonte-altezze alsometrie --file-alberi-calcolati $(CSV_DIR)/alberi-calcolati.csv

cd-per-compresa: analysis.py $(CSV_DIR)/alberi.csv $(CSV_DIR)/particelle.csv
	./analysis.py $(OTHER_FLAGS) --input-dir $(CSV_DIR) --genera-classi-diametriche --prefisso-output dati/per-compresa

cd-per-particella: analysis.py $(CSV_DIR)/alberi.csv $(CSV_DIR)/particelle.csv
	./analysis.py $(OTHER_FLAGS) --input-dir $(CSV_DIR) --genera-classi-diametriche --per-particella --prefisso-output dati/per-particella

ci-originali-per-compresa: analysis.py $(CSV_DIR)/alberi.csv $(CSV_DIR)/particelle.csv
	./analysis.py $(OTHER_FLAGS) --input-dir $(CSV_DIR) --genera-curve-ipsometriche --fonte-altezze originali --prefisso-output dati/per-compresa-originali

ci-originali-per-particella: analysis.py $(CSV_DIR)/alberi.csv $(CSV_DIR)/particelle.csv
	./analysis.py $(OTHER_FLAGS) --input-dir $(CSV_DIR) --genera-curve-ipsometriche --fonte-altezze originali --per-particella --prefisso-output dati/per-particella-originali

ci-interpolate-per-compresa: analysis.py $(CSV_DIR)/alberi.csv $(CSV_DIR)/particelle.csv $(CSV_DIR)/alsometrie.csv
	./analysis.py $(OTHER_FLAGS) --input-dir $(CSV_DIR) --genera-curve-ipsometriche --fonte-altezze ipsometro --ometti-generi-sconosciuti --prefisso-output dati/per-compresa-interpolate

ci-interpolate-per-particella: analysis.py $(CSV_DIR)/alberi.csv $(CSV_DIR)/particelle.csv $(CSV_DIR)/alsometrie.csv
	./analysis.py $(OTHER_FLAGS) --input-dir $(CSV_DIR) --genera-curve-ipsometriche --fonte-altezze ipsometro --ometti-generi-sconosciuti --per-particella --prefisso-output dati/per-particella-interpolate

$(CSV_DIR)/alberi-calcolati.csv:
	./acc.py --calcola-altezze --equazioni $(CSV_DIR)/equazioni_ipsometro.csv --input $(CSV_DIR)/alberi.csv --output $(CSV_DIR)/alberi-calcolati.csv

equazioni: $(CSV_DIR)/equazioni_originali.csv $(CSV_DIR)/equazioni_ipsometro.csv $(CSV_DIR)/equazioni_tabelle.csv

$(CSV_DIR)/equazioni_originali.csv: $(CSV_DIR)/alberi.csv
	./acc.py --genera-equazioni --funzione log --fonte-altezze originali --input $(CSV_DIR)/alberi.csv --output $(CSV_DIR)/equazioni_originali.csv

$(CSV_DIR)/equazioni_ipsometro.csv: $(CSV_DIR)/altezze.csv
	./acc.py --genera-equazioni --funzione log --fonte-altezze ipsometro --input $(CSV_DIR)/altezze.csv --output $(CSV_DIR)/equazioni_ipsometro.csv

$(CSV_DIR)/equazioni_tabelle.csv: $(CSV_DIR)/alsometrie.csv $(CSV_DIR)/particelle.csv
	./acc.py --genera-equazioni --funzione log --fonte-altezze tabelle --particelle $(CSV_DIR)/particelle.csv --input $(CSV_DIR)/alsometrie.csv --output $(CSV_DIR)/equazioni_tabelle.csv

html: html/per-compresa/classi-diametriche/index.html html/per-compresa/altezze-originali/index.html html/per-compresa/altezze-calcolate/index.html

html/per-compresa/classi-diametriche/index.html: $(CSV_DIR)/alberi.csv $(CSV_DIR)/particelle.csv $(CSV_DIR)/equazioni_ipsometro.csv template/html/per-compresa/classi-diametriche/index.html html/style.css
	./acc.py --report --formato html --equazioni $(CSV_DIR)/equazioni_ipsometro.csv --alberi $(CSV_DIR)/alberi.csv --particelle $(CSV_DIR)/particelle.csv --input template/html/per-compresa/classi-diametriche/index.html --output-dir html/per-compresa/classi-diametriche/

html/per-compresa/altezze-originali/index.html: $(CSV_DIR)/alberi.csv $(CSV_DIR)/particelle.csv $(CSV_DIR)/equazioni_originali.csv template/html/per-compresa/altezze-originali/index.html html/style.css
	./acc.py --report --formato html --equazioni $(CSV_DIR)/equazioni_ipsometro.csv --alberi $(CSV_DIR)/alberi.csv --particelle $(CSV_DIR)/particelle.csv --input template/html/per-compresa/altezze-originali/index.html --output-dir html/per-compresa/altezze-originali/

html/per-compresa/altezze-calcolate/index.html: $(CSV_DIR)/alberi.csv $(CSV_DIR)/particelle.csv $(CSV_DIR)/equazioni_ipsometro.csv template/html/per-compresa/altezze-calcolate/index.html html/style.css
	./acc.py --report --formato html --equazioni $(CSV_DIR)/equazioni_ipsometro.csv --alberi $(CSV_DIR)/alberi.csv --particelle $(CSV_DIR)/particelle.csv --input template/html/per-compresa/altezze-calcolate/index.html --output-dir html/per-compresa/altezze-calcolate/

dati: ./analysis.py $(CSV_DIR)/alberi.csv $(CSV_DIR)/alsometrie.csv $(CSV_DIR)/particelle.csv \
	$(CSV_DIR)/alsometrie-calcolate.csv $(CSV_DIR)/alberi-calcolati.csv \
	cd-per-compresa cd-per-particella \
	ci-originali-per-compresa ci-originali-per-particella \
	ci-interpolate-per-compresa ci-interpolate-per-particella

volumi: ./calcolo-volumi.py $(CSV_DIR)/alberi-calcolati.csv
	./calcolo-volumi.py $(CSV_DIR)/alberi-calcolati.csv -v

report: ./analysis.py $(CSV_DIR)/alberi.csv $(CSV_DIR)/alsometrie.csv $(CSV_DIR)/particelle.csv $(CSV_DIR)/altezze.csv
	./analysis.py --input-dir $(CSV_DIR) --un-genere-per-grafico --nome-report report --formato-output pdf --genera-classi-diametriche --genera-curve-ipsometriche --fonte-altezze ipsometro --ometti-generi-sconosciuti

sync:
	ssh laforesta.it "mkdir -p /var/www/laforestadotit/html/pdg-2026/accrescimenti"
	rsync -avz -e ssh . laforesta.it:/var/www/laforestadotit/html/pdg-2026/accrescimenti/

ipsometrie: ipsometrie.py $(CSV_DIR)/alberi-modello.csv $(CSV_DIR)/altezze.csv
	./ipsometrie.py $(CSV_DIR)/altezze.csv --fit log -o ipsometrie-log
	./ipsometrie.py $(CSV_DIR)/altezze.csv --fit lin -o ipsometrie-lin

clean-html:
	rm -f html/per-compresa/classi-diametriche/*  html/per-compresa/altezze-originali/*
