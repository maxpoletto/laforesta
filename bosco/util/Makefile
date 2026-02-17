.PHONY: find fetch precompute sync deploy

SENTINEL = ./sentinel.py

DATA_DIR = ../data
GEOJSON = $(DATA_DIR)/serra.geojson
OUTPUT_DIR = $(DATA_DIR)/satellite
DATES_FILE = $(OUTPUT_DIR)/dates.txt
FILES = --geojson $(GEOJSON) --output-dir $(OUTPUT_DIR)

MONTHS_SUMMER = 6,7
MAX_CLOUD_SUMMER = 1
DATES_SUMMER = $(OUTPUT_DIR)/dates-summer.txt

MONTHS_WINTER = 1,2
MAX_CLOUD_WINTER = 10
DATES_WINTER = $(OUTPUT_DIR)/dates-winter.txt

TARGET = laforesta.it:/var/www/laforestadotit/html/bosco/data/satellite

$(DATES_SUMMER):
	$(SENTINEL) find $(FILES) --pick $@ --months $(MONTHS_SUMMER) --overall-best --max-cloud $(MAX_CLOUD_SUMMER)

$(DATES_WINTER):
	$(SENTINEL) find $(FILES) --pick $@ --months $(MONTHS_WINTER) --overall-best --max-cloud $(MAX_CLOUD_WINTER)

$(DATES_FILE): $(DATES_SUMMER) $(DATES_WINTER)
	sort $^ >| $@

fetch: $(DATES_FILE)
	$(SENTINEL) fetch $(FILES) --dates-file $<

precompute:
	$(SENTINEL) precompute $(FILES)

sync:
      rsync -avz --delete -e ssh $(OUTPUT_DIR)/ $(TARGET)/

deploy: find fetch precompute sync

clean-satellite:
	rm -fr $(OUTPUT_DIR)
