.PHONY: all \
	cd-per-compresa cd-per-particella \
	ci-originali-per-compresa ci-originali-per-particella \
	ci-interpolate-per-compresa ci-interpolate-per-particella \
	dati sync clean ipsometrie

all: dati

alsometrie-calcolate.csv: analysis.py alsometrie.csv
	python3 analysis.py --genera-alsometrie-calcolate --metodo-altezze interpolazione --file-alsometrie-calcolate alsometrie-calcolate.csv

altezze-calcolate.csv: analysis.py alberi.csv alsometrie.csv
	python3 analysis.py --genera-alberi-altezze-calcolate --metodo-altezze interpolazione --file-alberi-calcolati alberi-calcolati.csv

cd-per-compresa: analysis.py alberi.csv particelle.csv
	python3 analysis.py --genera-classi-diametriche --prefisso-output dati/per-compresa

cd-per-particella: analysis.py alberi.csv particelle.csv
	python3 analysis.py --genera-classi-diametriche --per-particella --prefisso-output dati/per-particella

ci-originali-per-compresa: analysis.py alberi.csv particelle.csv
	python3 analysis.py --genera-curve-ipsometriche --metodo-altezze originali --prefisso-output dati/per-compresa-originali

ci-originali-per-particella: analysis.py alberi.csv particelle.csv
	python3 analysis.py --genera-curve-ipsometriche --metodo-altezze originali --per-particella --prefisso-output dati/per-particella-originali

ci-interpolate-per-compresa: analysis.py alberi.csv particelle.csv alsometrie.csv
	python3 analysis.py --genera-curve-ipsometriche --metodo-altezze interpolazione --ometti-generi-sconosciuti --prefisso-output dati/per-compresa-interpolate

ci-interpolate-per-particella: analysis.py alberi.csv particelle.csv alsometrie.csv
	python3 analysis.py --genera-curve-ipsometriche --metodo-altezze interpolazione --ometti-generi-sconosciuti --per-particella --prefisso-output dati/per-particella-interpolate

dati: analysis.py alberi.csv alsometrie.csv particelle.csv \
	alsometrie-calcolate.csv altezze-calcolate.csv \
	cd-per-compresa cd-per-particella \
	ci-originali-per-compresa ci-originali-per-particella \
	ci-interpolate-per-compresa ci-interpolate-per-particella

sync:
	ssh laforesta.it "mkdir -p /var/www/laforestadotit/html/pdg-2026/accrescimenti"
	rsync -avz -e ssh . laforesta.it:/var/www/laforestadotit/html/pdg-2026/accrescimenti/

ipsometrie: ipsometrie.py csv-alberi-modello.csv csv-altezze.csv
	python $< csv-alberi-modello.csv --fit lineare -o alberi1-lineare.png
	python $< csv-alberi-modello.csv --fit logaritmico -o alberi1-logaritmico.png
	python $< csv-altezze.csv --fit logaritmico -o alberi2-logaritmico.png
	python $< csv-altezze.csv --fit lineare -o alberi2-lineare.png

alsometrie.png: interpolate.py alsometrie.csv
	python $<

clean:
	rm -rf alberi-calcolati.csv alsometrie-calcolate.csv
	rm -rf alsometrie.png
	rm -rf alberi1-lineare.png alberi1-logaritmico.png alberi2-lineare.png alberi2-logaritmico.png
