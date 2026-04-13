PYTHON = python3
MANAGE = $(PYTHON) manage.py
BOSCO_DATA = ../bosco/data

.PHONY: check migrate import geo digest admin dev clean update-vendor minify test

check:
	$(MANAGE) check

migrate:
	mkdir -p data
	$(MANAGE) makemigrations
	$(MANAGE) migrate

import:
	$(PYTHON) ingest/import_reference.py
	$(PYTHON) ingest/import_parcels.py
	$(PYTHON) ingest/import_mannesi.py

geo:
	mkdir -p data/geo
	/bin/cp $(BOSCO_DATA)/particelle.geojson data/geo/
	/bin/cp $(BOSCO_DATA)/terreni.geojson data/geo/

digest:
	$(MANAGE) shell --no-imports -c "from apps.base.digests import generate_all; generate_all()"

admin:
	@$(MANAGE) shell --no-imports -c "from apps.base.models import User; exit(0 if User.objects.filter(is_superuser=True).exists() else 1)" \
		&& echo "Admin user already exists, skipping." \
		|| $(MANAGE) createsuperuser

dev: migrate import geo digest admin

clean:
	rm -rf data/
	find apps/*/migrations -name '*.py' ! -name '__init__.py' -delete

test:
	$(PYTHON) -m pytest --cov --cov-report=term-missing -q

collectstatic:
	$(MANAGE) collectstatic --noinput

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

# --- Minification -----------------------------------------------------------

JS_DIRS = apps/base/static/base/js \
          apps/prelievi/static/prelievi/js \
          apps/controllo/static/controllo/js \
          apps/impostazioni/static/impostazioni/js \
          apps/bosco/static/bosco/js

minify:
	@command -v terser >/dev/null 2>&1 || { echo "Install terser: npm i -g terser"; exit 1; }
	@for dir in $(JS_DIRS); do \
		for f in $$dir/*.js; do \
			[ -f "$$f" ] || continue; \
			echo "Minifying $$f"; \
			terser "$$f" --compress --mangle -o "$$f"; \
		done; \
	done
	@echo "Done. Run 'git checkout -- apps/*/static/*/js/' to restore sources."
