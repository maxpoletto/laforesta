PYTHON = python3
CANONICAL_DIR ?= data/canonical
SATELLITE_DIR ?= data/satellite
SATELLITE_GEOJSON ?= data/geo/terreni.geojson
SATELLITE_ENV_FILE ?= .env.satellite
BRANDING_DIR ?= data/branding
LOCAL_ENV_FILE ?= compose/.env.local
DJANGO_DEBUG ?= 1
DJANGO_SECRET_KEY ?= django-insecure-local-dev
LANGUAGE_SUFFIX ?= _it
STATIC_ROOT ?= staticfiles

REMOTE_USER ?= maxp
REMOTE_HOST ?= abies.laforesta.it
REMOTE ?= $(REMOTE_USER)@$(REMOTE_HOST)
REMOTE_ABIES_USER ?= abies
REMOTE_ABIES_GROUP ?= abies
ABIES_DOCKER_CONTEXT ?= vm-abies

ref ?=
REF ?= $(ref)
DEPLOY_ARGS ?=

LOCAL_ENV = set -a; [ ! -f "$(LOCAL_ENV_FILE)" ] || . "$(LOCAL_ENV_FILE)"; set +a;
SATELLITE_ENV = set -a; [ ! -f "$(SATELLITE_ENV_FILE)" ] || . "$(SATELLITE_ENV_FILE)"; set +a;
LOCAL_DJANGO_ENV = DJANGO_DEBUG="$${DJANGO_DEBUG:-$(DJANGO_DEBUG)}" DJANGO_SECRET_KEY="$${DJANGO_SECRET_KEY:-$(DJANGO_SECRET_KEY)}"
MANAGE = $(LOCAL_ENV) env $(LOCAL_DJANGO_ENV) $(PYTHON) manage.py
SATELLITE_MANAGE = $(LOCAL_ENV) $(SATELLITE_ENV) env $(LOCAL_DJANGO_ENV) $(PYTHON) manage.py

# Active language. Parallel to `from config.strings_it import *` in
# config/strings.py and `export * from './strings_it.js'` in strings.js:
# template files are committed as <name><LANGUAGE_SUFFIX>.html; the
# unsuffixed <name>.html symlinks are materialized at install time.

.PHONY: help \
        check migrate reset-db load-canonical-data bootstrap geo stage-marks-uploads digest admin link-templates sync-branding dev serve \
        satellite satellite-dates satellite-fetch satellite-precompute \
        collectstatic minify \
        deploy-dev deploy-prod deploy-instance \
        sync-canonical sync-canonical-dev sync-canonical-prod sync-satellite sync-satellite-dev sync-satellite-prod sync-data sync-data-dev sync-data-prod \
        bootstrap-dev bootstrap-prod bootstrap-instance backup-dev backup-prod backup-instance docker-prune \
        test test-js \
        clean reset-schema update-vendor

help:
	@echo "Local development:"
	@echo "  make dev                         reset local DB, load canonical data, create admin"
	@echo "  make serve                       run Django dev server in the background"
	@echo "  make test                        run JS tests and pytest"
	@echo "  make sync-branding                copy data/branding assets into static files"
	@echo "  make satellite                    build satellite dates, rasters, and summaries"
	@echo "  make satellite-precompute         rebuild satellite manifest/timeseries only"
	@echo "  make collectstatic minify        collect static files, then minify collected JS"
	@echo ""
	@echo "Remote deployment:"
	@echo "  make deploy-dev [REF=<ref>]      deploy current tree or ref to abies-dev"
	@echo "  make deploy-prod REF=<ref>       deploy clean tagged/ref build to production"
	@echo "  make sync-canonical-dev         rsync CANONICAL_DIR to dev data/canonical"
	@echo "  make sync-canonical-prod        rsync CANONICAL_DIR to prod data/canonical"
	@echo "  make sync-satellite-dev         rsync generated SATELLITE_DIR to dev"
	@echo "  make sync-satellite-prod        rsync generated SATELLITE_DIR to prod"
	@echo "  make sync-data-dev              rsync canonical + satellite data to dev"
	@echo "  make sync-data-prod             rsync canonical + satellite data to prod"
	@echo "  make bootstrap-dev              sync canonical data, then bootstrap dev DB"
	@echo "  make bootstrap-prod             sync canonical data, then bootstrap prod DB"
	@echo "  make backup-dev                  create and prune a dev backup"
	@echo "  make backup-prod                 create and prune a prod backup"
	@echo ""
	@echo "Useful variables: LOCAL_ENV_FILE=$(LOCAL_ENV_FILE), SATELLITE_ENV_FILE=$(SATELLITE_ENV_FILE), CANONICAL_DIR=$(CANONICAL_DIR), SATELLITE_DIR=$(SATELLITE_DIR), SATELLITE_GEOJSON=$(SATELLITE_GEOJSON), BRANDING_DIR=$(BRANDING_DIR), REMOTE=$(REMOTE), ABIES_DOCKER_CONTEXT=$(ABIES_DOCKER_CONTEXT), DEPLOY_ARGS=--reset-user"

# --- Local database and data build -----------------------------------------

check:
	$(MANAGE) check

migrate:
	mkdir -p data
	$(MANAGE) makemigrations
	$(MANAGE) migrate

# `make dev` runs `reset-db` first so bootstrap always loads into a clean DB.
reset-db:
	rm -f data/db.sqlite3 data/db.sqlite3-wal data/db.sqlite3-shm
	rm -rf data/ipso-inbox

load-canonical-data:
	mkdir -p data
	$(LOCAL_ENV) env $(LOCAL_DJANGO_ENV) PYTHON="$(PYTHON)" bin/load-canonical-data $(CANONICAL_DIR)

bootstrap:
	$(MANAGE) bootstrap $(CANONICAL_DIR)

geo:
	mkdir -p data/geo
	$(MANAGE) build_geo $(CANONICAL_DIR)

satellite:
	$(SATELLITE_MANAGE) build_satellite all --geojson "$(SATELLITE_GEOJSON)" --output-dir "$(SATELLITE_DIR)"

satellite-dates:
	$(SATELLITE_MANAGE) build_satellite dates --geojson "$(SATELLITE_GEOJSON)" --output-dir "$(SATELLITE_DIR)"

satellite-fetch:
	$(SATELLITE_MANAGE) build_satellite fetch --geojson "$(SATELLITE_GEOJSON)" --output-dir "$(SATELLITE_DIR)"

satellite-precompute:
	$(SATELLITE_MANAGE) build_satellite precompute --geojson "$(SATELLITE_GEOJSON)" --output-dir "$(SATELLITE_DIR)"

stage-marks-uploads:
	$(MANAGE) stage_marks_uploads $(CANONICAL_DIR)/marks

digest:
	$(MANAGE) shell --no-imports -c "from apps.base.digests import generate_all; generate_all()"

# Creates the initial superuser (role='admin'). Non-interactive when
# DJANGO_SUPERUSER_USERNAME/EMAIL/PASSWORD are all set; otherwise prompts.
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

dev: sync-branding reset-db load-canonical-data admin link-templates

serve:
	$(MANAGE) runserver > /dev/null 2>&1 &

# --- Static assets ----------------------------------------------------------

sync-branding:
	ABIES_BRANDING_DIR="$(BRANDING_DIR)" bin/sync-branding

collectstatic: sync-branding
	$(MANAGE) collectstatic --noinput

# Minifies collected static files only. It never rewrites app source files.
minify:
	bin/minify-static $(STATIC_ROOT)

# --- Remote deployment ------------------------------------------------------

deploy-dev:
	$(MAKE) deploy-instance INSTANCE=dev

deploy-prod:
	@test -n "$(REF)" || { echo "usage: make deploy-prod REF=<tag-or-sha> [DEPLOY_ARGS=--reset-user]" >&2; exit 1; }
	$(MAKE) deploy-instance INSTANCE=prod

deploy-instance:
	@test "$(INSTANCE)" = "dev" -o "$(INSTANCE)" = "prod" || { echo "usage: make deploy-instance INSTANCE=<dev|prod>" >&2; exit 1; }
	ABIES_DOCKER_CONTEXT="$(ABIES_DOCKER_CONTEXT)" ./bin/deploy $(INSTANCE) $(REF) $(DEPLOY_ARGS)

sync-canonical-dev:
	$(MAKE) sync-canonical INSTANCE=dev

sync-canonical-prod:
	$(MAKE) sync-canonical INSTANCE=prod

sync-canonical:
	@test -d "$(CANONICAL_DIR)" || { echo "error: CANONICAL_DIR='$(CANONICAL_DIR)' does not exist" >&2; exit 1; }
	@test "$(INSTANCE)" = "dev" -o "$(INSTANCE)" = "prod" || { echo "usage: make sync-canonical INSTANCE=<dev|prod>" >&2; exit 1; }
	ssh "$(REMOTE)" "sudo mkdir -p /var/lib/abies-$(INSTANCE)/data/canonical"
	rsync -rlt --delete --rsync-path="sudo rsync" "$(CANONICAL_DIR)/" "$(REMOTE):/var/lib/abies-$(INSTANCE)/data/canonical/"
	ssh "$(REMOTE)" "sudo chown -R $(REMOTE_ABIES_USER):$(REMOTE_ABIES_GROUP) /var/lib/abies-$(INSTANCE)/data/canonical"
	ssh "$(REMOTE)" "sudo chmod -R u=rwX,g=rX,o= /var/lib/abies-$(INSTANCE)/data/canonical"

sync-satellite-dev:
	$(MAKE) sync-satellite INSTANCE=dev

sync-satellite-prod:
	$(MAKE) sync-satellite INSTANCE=prod

sync-satellite:
	@test -d "$(SATELLITE_DIR)" || { echo "error: SATELLITE_DIR='$(SATELLITE_DIR)' does not exist" >&2; exit 1; }
	@test "$(INSTANCE)" = "dev" -o "$(INSTANCE)" = "prod" || { echo "usage: make sync-satellite INSTANCE=<dev|prod>" >&2; exit 1; }
	ssh "$(REMOTE)" "sudo mkdir -p /var/lib/abies-$(INSTANCE)/data/satellite"
	rsync -rlt --delete --rsync-path="sudo rsync" "$(SATELLITE_DIR)/" "$(REMOTE):/var/lib/abies-$(INSTANCE)/data/satellite/"
	ssh "$(REMOTE)" "sudo chown -R $(REMOTE_ABIES_USER):$(REMOTE_ABIES_GROUP) /var/lib/abies-$(INSTANCE)/data/satellite"
	ssh "$(REMOTE)" "sudo chmod -R u=rwX,g=rX,o= /var/lib/abies-$(INSTANCE)/data/satellite"

sync-data-dev:
	$(MAKE) sync-data INSTANCE=dev

sync-data-prod:
	$(MAKE) sync-data INSTANCE=prod

sync-data:
	@test "$(INSTANCE)" = "dev" -o "$(INSTANCE)" = "prod" || { echo "usage: make sync-data INSTANCE=<dev|prod>" >&2; exit 1; }
	$(MAKE) sync-canonical INSTANCE=$(INSTANCE)
	$(MAKE) sync-satellite INSTANCE=$(INSTANCE)

bootstrap-dev:
	$(MAKE) bootstrap-instance INSTANCE=dev

bootstrap-prod:
	$(MAKE) bootstrap-instance INSTANCE=prod

bootstrap-instance: sync-canonical
	@test "$(INSTANCE)" = "dev" -o "$(INSTANCE)" = "prod" || { echo "usage: make bootstrap-instance INSTANCE=<dev|prod>" >&2; exit 1; }
	./bin/bootstrap-data $(INSTANCE)

backup-dev:
	$(MAKE) backup-instance INSTANCE=dev

backup-prod:
	$(MAKE) backup-instance INSTANCE=prod

backup-instance:
	@test "$(INSTANCE)" = "dev" -o "$(INSTANCE)" = "prod" || { echo "usage: make backup-instance INSTANCE=<dev|prod>" >&2; exit 1; }
	docker --context "$(ABIES_DOCKER_CONTEXT)" exec "abies-$(INSTANCE)" bin/backup --instance "$(INSTANCE)"

docker-prune:
	docker --context "$(ABIES_DOCKER_CONTEXT)" image prune

# --- Tests ------------------------------------------------------------------

test: test-js
	$(LOCAL_ENV) env $(LOCAL_DJANGO_ENV) $(PYTHON) -m pytest --cov --cov-report=term-missing -q

# Pure-JS tests for shared client-side modules.
test-js:
	node apps/ipso/static/ipso/pwa/session.test.mjs
	node apps/ipso/static/ipso/pwa/csv.test.mjs
	node apps/ipso/static/ipso/pwa/upload.test.mjs
	node apps/ipso/static/ipso/pwa/store.test.mjs
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
	node apps/campionamenti/static/campionamenti/js/campionamenti.test.mjs
	node apps/piano_di_taglio/static/piano_di_taglio/js/piano-di-taglio.test.mjs

# --- Maintenance ------------------------------------------------------------

clean:
	rm -rf data/ staticfiles/
	find apps -type l -delete

# Dev-only: drop the auto-generated 0001_initial.py so the next `make migrate`
# regenerates it from the current model state. Once real data ships, stop using
# this target; model changes must then be incremental migrations.
reset-schema: clean
	find apps/*/migrations -name '0001_initial.py' -delete

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
