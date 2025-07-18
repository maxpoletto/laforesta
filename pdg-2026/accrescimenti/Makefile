.PHONY: all classi-diametriche curve-ipsometriche sync clean

all: classi-diametriche curve-ipsometriche

classi-diametriche:
	python3 analysis.py --solo-classi-diametriche

curve-ipsometriche:
	python3 analysis.py --solo-curve-ipsometriche

sync:
	rsync -avz -e ssh classi-diametriche/ laforesta.it:/var/www/laforestadotit/html/pdg-2026/classi-diametriche
	rsync -avz -e ssh curve-ipsometriche/ laforesta.it:/var/www/laforestadotit/html/pdg-2026/curve-ipsometriche

clean:
	rm -rf classi-diametriche
	rm -rf curve-ipsometriche
