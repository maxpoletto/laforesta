.PHONY: all clean

all: alsometrie

METODO_RIEMPIMENTO ?= fit
INTERPOLAZIONE ?= linear

alsometrie: interpola.py alsometrie.csv
	python $< -i alsometrie.csv -o alsometrie-calcolate.csv -p alsometrie.png --metodo-riempimento $(METODO_RIEMPIMENTO) --interpolazione $(INTERPOLAZIONE)

clean:
	rm -rf alsometrie_interpolate.csv alsometrie.png
