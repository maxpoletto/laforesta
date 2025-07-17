.PHONY: all histograms sync clean

all: histograms

histograms:
	python3 analysis.py

sync:
	rsync -avz -e ssh histograms/ laforesta.it:/var/www/laforestadotit/html/pdg-2026/classi-diametriche

clean:
	rm -rf histograms
