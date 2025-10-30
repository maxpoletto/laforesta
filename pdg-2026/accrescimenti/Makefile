.PHONY: all \
	cd-per-compresa cd-per-particella \
	ci-originali-per-compresa ci-originali-per-particella \
	ci-interpolate-per-compresa ci-interpolate-per-particella \
	dati sync clean

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

alberi-modello.png: alberi-modello.py
	python $<

alsometrie.png: interpolate.py alsometrie.csv
	python $<

clean:
	rm -rf alberi-calcolati.csv alsometrie-calcolate.csv
	rm -rf report
	rm -rf alsometrie.png
