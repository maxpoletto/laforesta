.PHONY: all report sync clean

all: report

alsometrie-calcolate.csv: analysis.py alsometrie.csv
	python3 analysis.py --genera-alsometrie-calcolate --metodo-altezze interpolazione --file-alsometrie-calcolate alsometrie-calcolate.csv

altezze-calcolate.csv: analysis.py alberi.csv alsometrie.csv
	python3 analysis.py --genera-alberi-altezze-calcolate --metodo-altezze interpolazione --file-alberi-calcolati alberi-calcolati.csv

originali-per-compresa: analysis.py alberi.csv particelle.csv
	python3 analysis.py --genera-classi-diametriche --genera-curve-ipsometriche --metodo-altezze originali --prefisso-output originali-per-compresa

originali-per-particella: analysis.py alberi.csv particelle.csv
	python3 analysis.py --genera-classi-diametriche --genera-curve-ipsometriche --metodo-altezze originali --per-particella --prefisso-output originali-per-particella

interpolate-per-compresa: analysis.py alberi.csv particelle.csv alsometrie.csv
	python3 analysis.py --genera-classi-diametriche --genera-curve-ipsometriche --metodo-altezze interpolazione --ometti-generi-sconosciuti --prefisso-output interpolate-per-compresa

interpolate-per-particella: analysis.py alberi.csv particelle.csv alsometrie.csv
	python3 analysis.py --genera-classi-diametriche --genera-curve-ipsometriche --metodo-altezze interpolazione --ometti-generi-sconosciuti --per-particella --prefisso-output interpolate-per-particella

report: analysis.py alberi.csv alsometrie.csv particelle.csv \
	alsometrie-calcolate.csv altezze-calcolate.csv \
	originali-per-compresa originali-per-particella \
	interpolate-per-compresa interpolate-per-particella

sync:
	ssh laforesta.it "mkdir -p /var/www/laforestadotit/html/pdg-2026/accrescimenti"
	rsync -avz -e ssh report/* laforesta.it:/var/www/laforestadotit/html/pdg-2026/accrescimenti/

alsometrie.png: interpolate.py alsometrie.csv
	python $<

clean:
	rm -rf alberi-calcolati.csv alsometrie-calcolate.csv
	rm -rf report
	rm -rf alsometrie.png
