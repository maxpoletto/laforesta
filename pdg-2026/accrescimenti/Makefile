.PHONY: all report sync clean

all: report

report: analysis.py alberi.csv alsometrie.csv particelle.csv
	python3 analysis.py --genera-alberi-altezze-calcolate --file-alberi-calcolati alberi-calcolati.csv
	python3 analysis.py --genera-alsometrie-calcolate --file-alsometrie-calcolate alsometrie-calcolate.csv
	python3 analysis.py --genera-classi-diametriche --genera-curve-ipsometriche --metodo-altezze originali --prefisso-output report/originali/per-compresa
	python3 analysis.py --genera-classi-diametriche --genera-curve-ipsometriche --metodo-altezze interpolazione --ometti-generi-sconosciuti --prefisso-output report/interpolazione/per-compresa
	python3 analysis.py --genera-classi-diametriche --genera-curve-ipsometriche --metodo-altezze originali --per-particella --prefisso-output report/originali/per-particella
	python3 analysis.py --genera-classi-diametriche --genera-curve-ipsometriche --metodo-altezze interpolazione --ometti-generi-sconosciuti --per-particella --prefisso-output report/interpolazione/per-particella

sync:
	ssh laforesta.it "mkdir -p /var/www/laforestadotit/html/pdg-2026/accrescimenti"
	rsync -avz -e ssh report/* laforesta.it:/var/www/laforestadotit/html/pdg-2026/accrescimenti/

alsometrie.png: interpolate.py alsometrie.csv
	python $<

clean:
	rm -rf alberi-calcolati.csv alsometrie-calcolate.csv
	rm -rf report
	rm -rf alsometrie.png
