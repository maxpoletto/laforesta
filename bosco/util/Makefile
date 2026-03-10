.PHONY: dates fetch precompute sync deploy test produzione calendario

SENTINEL = ./sentinel.py
PRODUCTION = ./production.py
CALENDARIO = ./calendario_mannesi.py

DATA_DIR = ../data
MANNESI_CSV = $(DATA_DIR)/mannesi.csv
GEOJSON = $(DATA_DIR)/terreni.geojson
SAT_DIR = $(DATA_DIR)/satellite

PROD_DIR = $(DATA_DIR)/produzione
PROD_CSV = $(DATA_DIR)/produzione-particelle-anno.csv

COMPRESE = Serra Fabrizia Capistrano

MONTHS_SUMMER = 6,7
MAX_CLOUD_SUMMER = 1

MONTHS_WINTER = 1,2
MAX_CLOUD_WINTER = 10

TARGET = laforesta.it:/var/www/laforestadotit/html/bosco/data

# Per-region output dirs: ../data/satellite/Serra/, ../data/satellite/Fabrizia/
# Each gets its own dates files, fetched bands, and precomputed data.

define REGION_RULES

$(SAT_DIR)/$(1)/dates-summer.txt:
	$(SENTINEL) find --geojson $(GEOJSON) --output-dir $(SAT_DIR)/$(1) --region $(1) \
		--pick $$@ --months $(MONTHS_SUMMER) --overall-best --max-cloud $(MAX_CLOUD_SUMMER)

$(SAT_DIR)/$(1)/dates-winter.txt:
	$(SENTINEL) find --geojson $(GEOJSON) --output-dir $(SAT_DIR)/$(1) --region $(1) \
		--pick $$@ --months $(MONTHS_WINTER) --overall-best --max-cloud $(MAX_CLOUD_WINTER)

$(SAT_DIR)/$(1)/dates.txt: $(SAT_DIR)/$(1)/dates-summer.txt $(SAT_DIR)/$(1)/dates-winter.txt
	sort $$^ >| $$@

.PHONY: dates-$(1) fetch-$(1) precompute-$(1)

dates-$(1): $(SAT_DIR)/$(1)/dates.txt

fetch-$(1): $(SAT_DIR)/$(1)/dates.txt
	$(SENTINEL) fetch --geojson $(GEOJSON) --output-dir $(SAT_DIR)/$(1) --region $(1) \
		--dates-file $$<

precompute-$(1):
	$(SENTINEL) precompute --geojson $(GEOJSON) --output-dir $(SAT_DIR)/$(1) --region $(1)

endef

$(foreach c,$(COMPRESE),$(eval $(call REGION_RULES,$(c))))

# --- Aggregate targets ---

dates: $(foreach c,$(COMPRESE),dates-$(c))

fetch: $(foreach c,$(COMPRESE),fetch-$(c))

precompute: $(foreach c,$(COMPRESE),precompute-$(c))

sync-satellite:
	rsync -avz --delete -e ssh $(SAT_DIR)/ $(TARGET)/satellite/

production:
	$(PRODUCTION) --geojson $(GEOJSON) --csv $(PROD_CSV) --output-dir $(PROD_DIR)

sync-production:
	rsync -avz --delete -e ssh $(PROD_DIR)/ $(TARGET)/produzione/

sync: sync-satellite sync-production

deploy: fetch precompute sync

clean-satellite:
	$(foreach c,$(COMPRESE),rm -f $(SAT_DIR)/$(c)/dates-summer.txt $(SAT_DIR)/$(c)/dates-winter.txt $(SAT_DIR)/$(c)/dates.txt;)

clean-all-satellite:
	rm -fr $(SAT_DIR)

clean-all-production:
	rm -fr $(PROD_DIR)

calendario: $(DATA_DIR)/calendario-mannesi.csv $(DATA_DIR)/calendario-mannesi-effettivo.csv

$(DATA_DIR)/calendario-mannesi.csv: $(MANNESI_CSV) $(DATA_DIR)/particelle.csv $(CALENDARIO)
	$(CALENDARIO) $(MANNESI_CSV) $@

$(DATA_DIR)/calendario-mannesi-effettivo.csv: $(MANNESI_CSV) $(CALENDARIO)
	$(CALENDARIO) --effettivo $(MANNESI_CSV) $@

test:
	./test_sentinel.py
	python -m pytest test_production.py test_calendario_mannesi.py
