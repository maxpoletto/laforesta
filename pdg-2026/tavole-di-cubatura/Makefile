.PHONY: all clean sync

all: alsometrie.png

alsometrie.png: alsometrie.csv
	python interpolate.py

sync: alsometrie.png
	ssh laforesta.it "cd /var/www/laforestadotit/html/pdg-2026 && mkdir -p tavole-di-cubatura"
	rsync -avz -e ssh alsometrie.png laforesta.it:/var/www/laforestadotit/html/pdg-2026/tavole-di-cubatura/

clean:
	rm -f alsometrie_interpolated.csv alsometrie.png
