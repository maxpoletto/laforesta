.PHONY: all classi-diametriche curve-ipsometriche sync clean

all: classi-diametriche curve-ipsometriche alsometrie.png

classi-diametriche:
	python3 analysis.py --solo-classi-diametriche

curve-ipsometriche:
	python3 analysis.py --solo-curve-ipsometriche

alsometrie.png: interpolate.py alsometrie.csv
	python $<

sync:
	rsync -avz -e ssh classi-diametriche/ laforesta.it:/var/www/laforestadotit/html/pdg-2026/classi-diametriche
	rsync -avz -e ssh curve-ipsometriche/ laforesta.it:/var/www/laforestadotit/html/pdg-2026/curve-ipsometriche
	rsync -avz -e ssh alsometrie.png laforesta.it:/var/www/laforestadotit/html/pdg-2026/

clean:
	rm -rf classi-diametriche
	rm -rf curve-ipsometriche
	rm -f alsometrie_interpolated.csv alsometrie.png
