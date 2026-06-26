.PHONY: all convert satellite

all: convert satellite

PYTHON ?= python3
SRC_DIR ?= ../abies-data
DST_DIR ?= ../abies/data/canonical
BRANDING_DST_DIR ?= ../abies/data/branding
SATELLITE_SRC_DIR ?= ../bosco/data/satellite
SATELLITE_DST_DIR ?= ../abies/data/satellite

convert:
	mkdir -p $(DST_DIR) $(BRANDING_DST_DIR)
	$(PYTHON) ./convert_laforesta.py $(SRC_DIR) $(DST_DIR) $(BRANDING_DST_DIR)

satellite:
	@test -d "$(SATELLITE_SRC_DIR)" || { echo "error: SATELLITE_SRC_DIR='$(SATELLITE_SRC_DIR)' does not exist" >&2; exit 1; }
	mkdir -p "$(SATELLITE_DST_DIR)"
	rsync -rlt --delete "$(SATELLITE_SRC_DIR)/" "$(SATELLITE_DST_DIR)/"
