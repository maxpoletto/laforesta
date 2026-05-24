PYTHON = python3
MANAGE = $(PYTHON) manage.py
BOSCO_DATA = ../bosco/data
PDG_CSV = ../pdg-2026/csv

# Active language.  Parallel to `from config.strings_it import *` in
# config/strings.py and `export * from './strings_it.js'` in strings.js:
# template files are committed as <name><LANGUAGE_SUFFIX>.html; the
# unsuffixed <name>.html symlinks are materialized at install time.
LANGUAGE_SUFFIX = _it

.PHONY: check migrate import import-base import-data geo digest admin dev \
        clean reset-schema update-vendor minify test test-js collectstatic \
        link-templates

check:
	$(MANAGE) check

migrate:
	mkdir -p data
	$(MANAGE) makemigrations
	$(MANAGE) migrate

import: import-base import-data

import-base:
	$(MANAGE) import_reference $(BOSCO_DATA)
	$(MANAGE) import_parcels $(BOSCO_DATA)

import-data:
	$(MANAGE) import_mannesi $(BOSCO_DATA)
	$(MANAGE) import_sample_grid $(BOSCO_DATA)
	$(MANAGE) import_sampled_trees $(BOSCO_DATA)
	$(MANAGE) import_pai $(BOSCO_DATA)
	$(MANAGE) import_calendar $(PDG_CSV)

geo:
	mkdir -p data/geo
	/bin/cp $(BOSCO_DATA)/particelle.geojson data/geo/
	/bin/cp $(BOSCO_DATA)/terreni.geojson data/geo/

digest:
	$(MANAGE) shell --no-imports -c "from apps.base.digests import generate_all; generate_all()"

admin:
	@$(MANAGE) shell --no-imports -c "from apps.base.models import User; exit(0 if User.objects.filter(is_superuser=True).exists() else 1)" \
		&& echo "Admin user already exists, skipping." \
		|| ( $(MANAGE) createsuperuser && \
		     $(MANAGE) shell --no-imports -c "from apps.base.models import User; u = User.objects.get(is_superuser=True); u.role = 'admin'; u.save(update_fields=['role'])" )

link-templates:
	$(PYTHON) bin/link-templates.py $(LANGUAGE_SUFFIX)

dev: migrate import geo digest admin link-templates

dev-nodata: migrate import-base geo digest admin link-templates

serve:
	$(PYTHON) manage.py runserver > /dev/null 2>&1 &

clean:
	rm -rf data/
	find apps -type l -delete

# Dev-only: drop the auto-generated 0001_initial.py so the next
# `make migrate` regenerates it from the current model state.  Use this
# during early development when the schema is still churning and you'd
# rather collapse model edits into a fresh initial than accumulate
# incremental migrations.  Hand-written migrations (0002_triggers.py,
# future data fixes, etc.) are preserved.  Once the schema is shipped
# with real data, stop using this target — every model change must
# then be an incremental migration.
reset-schema: clean
	find apps/*/migrations -name '0001_initial.py' -delete

test: test-js
	$(PYTHON) -m pytest --cov --cov-report=term-missing -q

# Pure-JS tests for shared client-side modules (e.g. geo.js).
test-js:
	node test/test_geo.mjs

collectstatic:
	$(MANAGE) collectstatic --noinput

deploy-dev:
	./bin/deploy dev

bootstrap-dev:
	./bin/bootstrap-data dev

docker-prune:
	docker --context vm-abies image prune

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
	@echo "Vendor JS/CSS updated. Roboto fonts are managed manually."

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
