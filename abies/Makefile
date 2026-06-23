PYTHON = python3
CANONICAL_DIR ?= data/canonical
DJANGO_DEBUG ?= 1
DJANGO_SECRET_KEY ?= django-insecure-local-dev
MANAGE = env DJANGO_DEBUG="$(DJANGO_DEBUG)" DJANGO_SECRET_KEY="$(DJANGO_SECRET_KEY)" $(PYTHON) manage.py

# Active language.  Parallel to `from config.strings_it import *` in
# config/strings.py and `export * from './strings_it.js'` in strings.js:
# template files are committed as <name><LANGUAGE_SUFFIX>.html; the
# unsuffixed <name>.html symlinks are materialized at install time.
LANGUAGE_SUFFIX = _it

.PHONY: check migrate geo digest admin dev \
        clean reset-schema update-vendor minify test test-js collectstatic \
        link-templates reset-db bootstrap stage-marks-uploads

check:
	$(MANAGE) check

migrate:
	mkdir -p data
	$(MANAGE) makemigrations
	$(MANAGE) migrate

# --- Canonical load path: load canonical CSVs into an EMPTY instance.
# `make dev` runs `reset-db` first so bootstrap always loads into a clean DB.
reset-db:
	rm -f data/db.sqlite3 data/db.sqlite3-wal data/db.sqlite3-shm
	rm -rf data/ipso-inbox

bootstrap:
	$(MANAGE) bootstrap $(CANONICAL_DIR)

geo:
	mkdir -p data/geo
	$(MANAGE) build_geo $(CANONICAL_DIR)

stage-marks-uploads:
	$(MANAGE) stage_marks_uploads $(CANONICAL_DIR)/marks

digest:
	$(MANAGE) shell --no-imports -c "from apps.base.digests import generate_all; generate_all()"

# Creates the initial superuser (role='admin').  Non-interactive when
# DJANGO_SUPERUSER_USERNAME/EMAIL/PASSWORD are all set (so a reset-based
# `make dev` can run unattended); falls back to an interactive prompt otherwise.
admin:
	@$(MANAGE) shell --no-imports -c "from apps.base.models import User; exit(0 if User.objects.filter(is_superuser=True).exists() else 1)" \
		&& echo "Admin user already exists, skipping." \
		|| ( if [ -n "$$DJANGO_SUPERUSER_USERNAME" ] && [ -n "$$DJANGO_SUPERUSER_EMAIL" ] && [ -n "$$DJANGO_SUPERUSER_PASSWORD" ]; then \
		         $(MANAGE) createsuperuser --noinput; \
		     else \
		         $(MANAGE) createsuperuser; \
		     fi && \
		     $(MANAGE) shell --no-imports -c "from apps.base.models import User; u = User.objects.get(is_superuser=True); u.role = 'admin'; u.save(update_fields=['role'])" )

link-templates:
	$(PYTHON) bin/link-templates.py $(LANGUAGE_SUFFIX)

dev: reset-db migrate bootstrap geo stage-marks-uploads digest admin link-templates

serve:
	$(MANAGE) runserver > /dev/null 2>&1 &

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
	env DJANGO_DEBUG="$(DJANGO_DEBUG)" DJANGO_SECRET_KEY="$(DJANGO_SECRET_KEY)" $(PYTHON) -m pytest --cov --cov-report=term-missing -q

# Pure-JS tests for shared client-side modules (e.g. geo.js).
test-js:
	node apps/ipso/static/ipso/pwa/session.test.mjs
	node apps/base/static/base/js/geo.test.mjs
	node apps/base/static/base/js/csv-export.test.mjs
	node apps/base/static/base/js/charts.test.mjs
	node apps/base/static/base/js/coppice.test.mjs
	node apps/base/static/base/js/digests.test.mjs
	node apps/base/static/base/js/templates.test.mjs
	node apps/base/static/base/js/forms.test.mjs
	node apps/base/static/base/js/ui-widgets-modals.test.mjs
	node apps/base/static/base/js/tests.mjs
	node apps/base/static/base/js/cache.test.mjs
	node apps/base/static/base/js/page-sync.test.mjs
	node --check apps/bosco/static/bosco/js/bosco.js
	node apps/bosco/static/bosco/js/bosco-state.test.mjs
	node apps/bosco/static/bosco/js/bosco-satellite.test.mjs
	node apps/bosco/static/bosco/js/bosco-production.test.mjs
	node apps/bosco/static/bosco/js/bosco-characteristics.test.mjs
	node apps/bosco/static/bosco/js/bosco-detail.test.mjs
	node apps/bosco/static/bosco/js/bosco-pai.test.mjs
	node apps/mannesi/static/squadre/js/pdf.test.mjs
	node apps/mannesi/static/squadre/js/squadre.test.mjs
	node apps/prelievi/static/prelievi/js/prelievi.test.mjs
	node apps/base/static/base/js/prelievi-columns.test.mjs
	node apps/base/static/base/js/parcel-map.test.mjs
	node apps/base/static/base/js/map-tools.test.mjs

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
	npm install leaflet@1.9.4 chart.js@4.5.1 @maxpoletto/sortable-table@1.3.0
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
