.PHONY: all

all: convert

PYTHON ?= python3
SRC_DIR ?= ../abies-data
DST_DIR ?= ../abies/data/canonical

convert:
	mkdir -p $(DST_DIR)
	$(PYTHON) ./convert_laforesta.py $(SRC_DIR) $(DST_DIR)
