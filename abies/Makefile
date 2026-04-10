PYTHON = python3
MANAGE = $(PYTHON) manage.py

.PHONY: check migrate import digest admin dev update-vendor

check:
	$(MANAGE) check

migrate:
	$(MANAGE) makemigrations
	$(MANAGE) migrate

import:
	$(PYTHON) ingest/import_reference.py
	$(PYTHON) ingest/import_parcels.py
	$(PYTHON) ingest/import_mannesi.py

digest:
	$(MANAGE) shell -c "from apps.base.digests import generate_all; generate_all()"

admin:
	$(MANAGE) createsuperuser

dev: migrate import digest admin

# --- Vendor dependencies ----------------------------------------------------

VENDOR = apps/base/static/vendor
NPM_TMP = /tmp/abies-vendor

update-vendor:
	mkdir -p $(NPM_TMP) && cd $(NPM_TMP) && npm init -y --silent && \
	npm install leaflet@1.9.4 chart.js@4.5.1 @maxpoletto/sortable-table@1.0.4
	/bin/cp $(NPM_TMP)/node_modules/leaflet/dist/leaflet.js $(VENDOR)/leaflet/
	/bin/cp $(NPM_TMP)/node_modules/leaflet/dist/leaflet.css $(VENDOR)/leaflet/
	/bin/cp -r $(NPM_TMP)/node_modules/leaflet/dist/images $(VENDOR)/leaflet/
	/bin/cp $(NPM_TMP)/node_modules/chart.js/dist/chart.umd.js $(VENDOR)/chartjs/
	/bin/cp $(NPM_TMP)/node_modules/@maxpoletto/sortable-table/src/sortable-table.js $(VENDOR)/sortable-table/
	/bin/cp $(NPM_TMP)/node_modules/@maxpoletto/sortable-table/src/sortable-table.css $(VENDOR)/sortable-table/
	@echo "Vendor JS/CSS updated. DM Sans fonts are managed manually."
