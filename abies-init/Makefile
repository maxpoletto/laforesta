.PHONY: all convert

all: convert

PYTHON ?= python3
SRC_DIR ?= ../abies-data
DST_DIR ?= ../abies/data/canonical
BRANDING_DST_DIR ?= ../abies/data/branding

convert:
	mkdir -p $(DST_DIR) $(BRANDING_DST_DIR)
	$(PYTHON) ./convert_laforesta.py $(SRC_DIR) $(DST_DIR) $(BRANDING_DST_DIR)
