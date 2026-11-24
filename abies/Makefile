PYTHON = python3
CANONICAL_DIR ?= data/canonical
SATELLITE_DIR ?= data/satellite
MAP_GEOJSON ?= $(CANONICAL_DIR)
SATELLITE_GEOJSON ?=
MAP_CONTEXT_DIR ?= data/map-context
MAP_CONTEXT_GEOJSON ?=
MAP_CONTEXT_ARGS ?=
SATELLITE_ENV_FILE ?= .env.satellite
BRANDING_DIR ?= data/branding
LOCAL_ENV_FILE ?= compose/.env.local
LOCAL_PORT ?= 8000
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
MIRROR_ARGS ?=
SYNC_ARGS ?=

LOCAL_ENV = set -a; [ ! -f "$(LOCAL_ENV_FILE)" ] || . "$(LOCAL_ENV_FILE)"; set +a;
SATELLITE_ENV = set -a; [ ! -f "$(SATELLITE_ENV_FILE)" ] || . "$(abspath $(SATELLITE_ENV_FILE))"; set +a;
LOCAL_DJANGO_ENV = DJANGO_DEBUG="$${DJANGO_DEBUG:-$(DJANGO_DEBUG)}" DJANGO_SECRET_KEY="$${DJANGO_SECRET_KEY:-$(DJANGO_SECRET_KEY)}"
MANAGE = $(LOCAL_ENV) env $(LOCAL_DJANGO_ENV) $(PYTHON) manage.py
SATELLITE_MANAGE = $(LOCAL_ENV) $(SATELLITE_ENV) env $(LOCAL_DJANGO_ENV) $(PYTHON) manage.py

# Active language. Parallel to `from config.strings_it import *` in
# config/strings.py and `export * from './strings_it.js'` in strings.js:
# template files are committed as <name><LANGUAGE_SUFFIX>.html; the
# unsuffixed <name>.html symlinks are materialized at install time.

.PHONY: help \
        local run-local stop-local deploy-local reset-local mirror-prod-to-local sync-prod-to-local \
        serve terrain-worker check migrate load-canonical-data bootstrap geo stage-marks-uploads digest admin link-templates \
        test test-js test-print-js-coverage \
        deploy-dev deploy-prod deploy-instance backup-dev backup-prod backup-instance \
        mirror-prod-to-dev mirror-instance bootstrap-dev bootstrap-prod bootstrap-instance \
        sync-canonical-dev sync-canonical-prod sync-canonical \
        sync-branding collectstatic minify update-vendor check-vendor \
        lock-dependencies upgrade-dependencies audit-dependencies clean reset-schema docker-prune \
        map-context sync-map-context-dev sync-map-context-prod sync-map-context \
        satellite satellite-dates satellite-fetch satellite-precompute \
        sync-satellite-dev sync-satellite-prod sync-satellite sync-data-dev sync-data-prod sync-data

help:
	@echo "Local development"
	@echo "  make local                           Apply local migrations and link templates"
	@echo "  make run-local [LOCAL_PORT=8000]     Run Django and the terrain worker"
	@echo "  make stop-local                      Stop this checkout's local processes"
	@echo "  make deploy-local                    Prepare local setup and start both processes"
	@echo "  make reset-local                     Replace local database and inbox from CANONICAL_DIR"
	@echo "  make mirror-prod-to-local            Replace stopped local state from production (MIRROR_ARGS=--yes)"
	@echo "  make geo [MAP_GEOJSON=path]          Import an initial parcel map"
	@echo "  make test                            Run JavaScript tests and pytest"
	@echo "  make test-print-js-coverage          Report JavaScript print coverage"
	@echo ""
	@echo "Hosted instances"
	@echo "  make deploy-dev [REF=<ref>]          Deploy the current tree or a ref to abies-dev"
	@echo "  make deploy-prod REF=<ref>           Deploy a ref from a clean checkout to abies-prod"
	@echo "  make backup-dev / backup-prod        Create and prune an instance backup"
	@echo "  make mirror-prod-to-dev              Replace dev state from production (MIRROR_ARGS=--yes)"
	@echo "  make bootstrap-dev / bootstrap-prod"
	@echo "                                       Copy canonical data and initialize the database"
	@echo "  make sync-canonical-dev / sync-canonical-prod"
	@echo "                                       Copy CANONICAL_DIR to the instance"
	@echo ""
	@echo "Other operations"
	@echo "  make sync-branding                   Copy assets from BRANDING_DIR"
	@echo "  make collectstatic minify            Collect static files and minify collected JavaScript"
	@echo "  make update-vendor / check-vendor    Copy or verify locked browser assets"
	@echo "  make lock-dependencies               Regenerate Python locks on Python 3.14"
	@echo "  make upgrade-dependencies            Upgrade Python locks within approved ranges"
	@echo "  make audit-dependencies              Audit locked Python dependencies"
	@echo ""
	@echo "  Printable-map context (manual)"
	@echo "  make map-context                     Build roads, water and elevation contours"
	@echo "  make sync-map-context-dev / sync-map-context-prod"
	@echo "                                       Copy the generated MAP_CONTEXT_DIR"
	@echo ""
	@echo "  Standalone satellite data (legacy)"
	@echo "  make satellite                       Build dates, rasters and summaries using SATELLITE_ENV_FILE"
	@echo "  make satellite-precompute            Rebuild the manifest and parcel statistics"
	@echo "  make sync-satellite-dev / sync-satellite-prod"
	@echo "                                       Copy standalone rasters from SATELLITE_DIR"
	@echo "  make sync-data-dev / sync-data-prod"
	@echo "                                       Copy canonical data and standalone satellite rasters"

# --- Local development -----------------------------------------------------

local: migrate link-templates

run-local:
	$(LOCAL_ENV) env $(LOCAL_DJANGO_ENV) $(PYTHON) bin/run-local.py "$(LOCAL_PORT)"

stop-local:
	$(PYTHON) bin/run-local.py --stop

deploy-local: sync-branding local
	$(MAKE) run-local

# Run this target directly when replacing the local runtime data is intended.
reset-local: export ABIES_LEGACY_GEO_DIR =
reset-local: sync-branding link-templates
	@test -d "$(CANONICAL_DIR)" || { echo "error: CANONICAL_DIR='$(CANONICAL_DIR)' does not exist" >&2; exit 1; }
	rm -f data/db.sqlite3 data/db.sqlite3-wal data/db.sqlite3-shm
	rm -rf data/ipso-inbox
	$(MAKE) load-canonical-data
	$(MAKE) admin

mirror-prod-to-local: link-templates
	$(LOCAL_ENV) env $(LOCAL_DJANGO_ENV) ABIES_DOCKER_CONTEXT="$(ABIES_DOCKER_CONTEXT)" $(PYTHON) bin/sync-prod-to-local.py $(or $(MIRROR_ARGS),$(SYNC_ARGS))

sync-prod-to-local: mirror-prod-to-local

geo:
	$(MANAGE) build_geo "$(MAP_GEOJSON)"

# Tests

test:
	@$(LOCAL_ENV) $(PYTHON) -c 'import pytest, pytest_cov, pytest_django, django; from geographiclib.geodesic import Geodesic' || { \
		printf '%s\n' 'Install the locked development dependencies for the Python interpreter used by make.' \
			'  $(PYTHON) -m pip install --require-hashes -r requirements-dev.txt' \
			'See README.md for setting up a fresh Python 3.14 virtual environment.' >&2; \
		exit 1; \
	}
	$(MAKE) test-js
	$(LOCAL_ENV) env $(LOCAL_DJANGO_ENV) $(PYTHON) -m pytest --cov --cov-report=term-missing -q

test-print-js-coverage:
	node test/print-coverage.mjs

# Pure-JS tests for shared client-side modules.
test-js:
	node apps/impostazioni/static/impostazioni/js/drawing-parts.test.mjs
	node apps/impostazioni/static/impostazioni/js/draft-edits.test.mjs
	node apps/base/static/base/js/geodesy.test.mjs
	node apps/base/static/base/js/metric-geometry.test.mjs
	node apps/ipso/static/ipso/pwa/ipso.test.mjs
	node apps/base/static/base/js/equations.test.mjs
	node apps/controllo/static/controllo/js/columns.test.mjs
	node apps/base/static/base/js/api.test.mjs
	node apps/impostazioni/static/impostazioni/js/species-types.test.mjs
	node apps/impostazioni/static/impostazioni/js/entity-form.test.mjs
	node apps/impostazioni/static/impostazioni/js/species-equations.test.mjs
	node apps/impostazioni/static/impostazioni/js/equation-imports.test.mjs
	node apps/ipso/static/ipso/pwa/session.test.mjs
	node apps/ipso/static/ipso/pwa/modes.test.mjs
	node apps/ipso/static/ipso/pwa/csv.test.mjs
	node apps/ipso/static/ipso/pwa/gps.test.mjs
	node apps/ipso/static/ipso/pwa/point-recorder.test.mjs
	node apps/ipso/static/ipso/pwa/format.test.mjs
	node apps/ipso/static/ipso/pwa/map.test.mjs
	node apps/ipso/static/ipso/pwa/photo.test.mjs
	node apps/ipso/static/ipso/pwa/upload.test.mjs
	node apps/ipso/static/ipso/pwa/store.test.mjs
	node apps/ipso/static/ipso/pwa/app.test.mjs
	node apps/ipso/static/ipso/js/importazione.test.mjs
	node apps/ipso/static/ipso/js/import-review.test.mjs
	node apps/base/static/base/js/geo.test.mjs
	node apps/base/static/base/js/format.test.mjs
	node apps/base/static/base/js/csv-export.test.mjs
	node apps/base/static/base/js/tree-form.test.mjs
	node apps/base/static/base/js/charts.test.mjs
	node apps/base/static/base/js/dendrometry.test.mjs
	node apps/base/static/base/js/tree-detail.test.mjs
	node apps/base/static/base/js/coppice.test.mjs
	node apps/base/static/base/js/digests.test.mjs
	node apps/base/static/base/js/templates.test.mjs
	node apps/base/static/base/js/forms.test.mjs
	node apps/base/static/base/js/forms-csv-modal.test.mjs
	node apps/base/static/base/js/ui-widgets-modals.test.mjs
	node apps/base/static/base/js/sections.test.mjs
	node apps/base/static/base/js/tests.mjs
	node apps/base/static/base/js/cache.test.mjs
	node apps/base/static/base/js/router.test.mjs
	node apps/base/static/base/js/page-sync.test.mjs
	node apps/base/static/base/js/table-selection.test.mjs
	node apps/base/static/base/js/table-data.test.mjs
	node apps/base/static/base/js/table-delete.test.mjs
	node --check apps/bosco/static/bosco/js/bosco.js
	node apps/bosco/static/bosco/js/bosco-plan-source.test.mjs
	node apps/bosco/static/bosco/js/bosco-calendar.test.mjs
	node apps/bosco/static/bosco/js/characteristic-print.test.mjs
	node apps/bosco/static/bosco/js/bosco-metric-print.test.mjs
	node apps/bosco/static/bosco/js/bosco-print-ui.test.mjs
	node apps/bosco/static/bosco/js/bosco-print-context.test.mjs
	node apps/bosco/static/bosco/js/bosco-state.test.mjs
	node apps/bosco/static/bosco/js/bosco-satellite.test.mjs
	node apps/bosco/static/bosco/js/bosco-production.test.mjs
	node apps/bosco/static/bosco/js/bosco-characteristics.test.mjs
	node apps/bosco/static/bosco/js/bosco-detail.test.mjs
	node apps/bosco/static/bosco/js/bosco-pai.test.mjs
	node apps/bosco/static/bosco/js/bosco-observations.test.mjs
	node apps/bosco/static/bosco/js/observation-pdf.test.mjs
	node apps/mannesi/static/squadre/js/pdf.test.mjs
	node apps/mannesi/static/squadre/js/squadre.test.mjs
	node apps/prelievi/static/prelievi/js/prelievi.test.mjs
	node apps/base/static/base/js/prelievi-columns.test.mjs
	node apps/base/static/base/js/map-palette.test.mjs
	node apps/base/static/base/js/map-common.test.mjs
	node apps/base/static/base/js/parcel-map.test.mjs
	node apps/base/static/base/js/tree-points-map.test.mjs
	node apps/base/static/base/js/map-tools.test.mjs
	node apps/campionamenti/static/campionamenti/js/grid-planner.test.mjs
	node apps/campionamenti/static/campionamenti/js/campionamenti.test.mjs
	node apps/piano_di_taglio/static/piano_di_taglio/js/piano-di-taglio.test.mjs

# Local setup and process helpers

serve:
	$(MANAGE) runserver > /dev/null 2>&1 &

terrain-worker:
	$(MANAGE) run_terrain_worker

check:
	$(MANAGE) check

migrate:
	mkdir -p data
	$(MANAGE) makemigrations
	$(MANAGE) migrate

load-canonical-data:
	mkdir -p data
	$(LOCAL_ENV) env $(LOCAL_DJANGO_ENV) PYTHON="$(PYTHON)" bin/load-canonical-data $(CANONICAL_DIR)

bootstrap:
	$(MANAGE) bootstrap $(CANONICAL_DIR)

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

# --- Hosted instances ------------------------------------------------------

deploy-dev:
	$(MAKE) deploy-instance INSTANCE=dev

deploy-prod:
	@test -n "$(REF)" || { echo "usage: make deploy-prod REF=<tag-or-sha> [DEPLOY_ARGS=--reset-user]" >&2; exit 1; }
	$(MAKE) deploy-instance INSTANCE=prod

deploy-instance:
	@test "$(INSTANCE)" = "dev" -o "$(INSTANCE)" = "prod" || { echo "usage: make deploy-instance INSTANCE=<dev|prod>" >&2; exit 1; }
	ABIES_DOCKER_CONTEXT="$(ABIES_DOCKER_CONTEXT)" ./bin/deploy $(INSTANCE) $(REF) $(DEPLOY_ARGS)

backup-dev:
	$(MAKE) backup-instance INSTANCE=dev

backup-prod:
	$(MAKE) backup-instance INSTANCE=prod

backup-instance:
	@test "$(INSTANCE)" = "dev" -o "$(INSTANCE)" = "prod" || { echo "usage: make backup-instance INSTANCE=<dev|prod>" >&2; exit 1; }
	docker --context "$(ABIES_DOCKER_CONTEXT)" exec "abies-$(INSTANCE)" bin/backup --instance "$(INSTANCE)"

mirror-prod-to-dev:
	$(MAKE) mirror-instance SOURCE=prod TARGET=dev

mirror-instance:
	@test "$(SOURCE)" = "dev" -o "$(SOURCE)" = "prod" || { echo "usage: make mirror-instance SOURCE=<dev|prod> TARGET=<dev|prod> [MIRROR_ARGS=--yes]" >&2; exit 1; }
	@test "$(TARGET)" = "dev" -o "$(TARGET)" = "prod" || { echo "usage: make mirror-instance SOURCE=<dev|prod> TARGET=<dev|prod> [MIRROR_ARGS=--yes]" >&2; exit 1; }
	ABIES_DOCKER_CONTEXT="$(ABIES_DOCKER_CONTEXT)" ./bin/mirror-instance-state "$(SOURCE)" "$(TARGET)" $(MIRROR_ARGS)

bootstrap-dev:
	$(MAKE) bootstrap-instance INSTANCE=dev

bootstrap-prod:
	$(MAKE) bootstrap-instance INSTANCE=prod

bootstrap-instance: sync-canonical
	@test "$(INSTANCE)" = "dev" -o "$(INSTANCE)" = "prod" || { echo "usage: make bootstrap-instance INSTANCE=<dev|prod>" >&2; exit 1; }
	./bin/bootstrap-data $(INSTANCE)

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

# --- Other operations ------------------------------------------------------

sync-branding:
	ABIES_BRANDING_DIR="$(BRANDING_DIR)" bin/sync-branding

collectstatic: sync-branding
	$(MANAGE) collectstatic --noinput

# Minifies collected static files only. It never rewrites app source files.
minify:
	bin/minify-static $(STATIC_ROOT)

update-vendor:
	npm ci --ignore-scripts --prefix vendor
	$(PYTHON) bin/update-vendor
	@echo "Vendor JS/CSS updated. Roboto fonts are managed manually."

check-vendor:
	$(PYTHON) bin/update-vendor --check

# Use the same interpreter as production when resolving conditional dependencies.
lock-dependencies:
	$(PYTHON) -c 'import sys; assert sys.version_info[:2] == (3, 14), "Compile locks with Python 3.14"'
	$(PYTHON) -m uv pip compile --python "$(PYTHON)" --generate-hashes --default-index https://pypi.org/simple --custom-compile-command 'make lock-dependencies' $(UV_COMPILE_UPGRADE) --output-file requirements.txt requirements.in
	$(PYTHON) -m uv pip compile --python "$(PYTHON)" --generate-hashes --default-index https://pypi.org/simple --custom-compile-command 'make lock-dependencies' $(UV_COMPILE_UPGRADE) --output-file requirements-dev.txt requirements-dev.in

upgrade-dependencies:
	$(MAKE) lock-dependencies PYTHON="$(PYTHON)" UV_COMPILE_UPGRADE=--upgrade

audit-dependencies:
	$(PYTHON) -m pip_audit --strict --disable-pip --require-hashes -r requirements-dev.txt

clean:
	rm -rf data/ staticfiles/
	find apps -type l -delete

# Dev-only: drop the auto-generated 0001_initial.py so the next `make migrate`
# regenerates it from the current model state. Once real data ships, stop using
# this target; model changes must then be incremental migrations.
reset-schema: clean
	find apps/*/migrations -name '0001_initial.py' -delete

docker-prune:
	docker --context "$(ABIES_DOCKER_CONTEXT)" image prune

# Printable-map context (manual)

map-context:
	$(MANAGE) build_map_context $(if $(MAP_CONTEXT_GEOJSON),--geojson "$(MAP_CONTEXT_GEOJSON)") --output-dir "$(MAP_CONTEXT_DIR)" $(MAP_CONTEXT_ARGS)

sync-map-context-dev:
	$(MAKE) sync-map-context INSTANCE=dev

sync-map-context-prod:
	$(MAKE) sync-map-context INSTANCE=prod

sync-map-context:
	@test -d "$(MAP_CONTEXT_DIR)" || { echo "error: MAP_CONTEXT_DIR='$(MAP_CONTEXT_DIR)' does not exist" >&2; exit 1; }
	@test "$(INSTANCE)" = "dev" -o "$(INSTANCE)" = "prod" || { echo "usage: make sync-map-context INSTANCE=<dev|prod>" >&2; exit 1; }
	ssh "$(REMOTE)" "sudo mkdir -p /var/lib/abies-$(INSTANCE)/data/map-context"
	rsync -rlt --delete --rsync-path="sudo rsync" "$(MAP_CONTEXT_DIR)/" "$(REMOTE):/var/lib/abies-$(INSTANCE)/data/map-context/"
	ssh "$(REMOTE)" "sudo chown -R $(REMOTE_ABIES_USER):$(REMOTE_ABIES_GROUP) /var/lib/abies-$(INSTANCE)/data/map-context"
	ssh "$(REMOTE)" "sudo chmod -R u=rwX,g=rX,o= /var/lib/abies-$(INSTANCE)/data/map-context"

# Standalone satellite data (legacy)

satellite:
	$(SATELLITE_MANAGE) build_satellite all $(if $(SATELLITE_GEOJSON),--geojson "$(SATELLITE_GEOJSON)") --output-dir "$(SATELLITE_DIR)"

satellite-dates:
	$(SATELLITE_MANAGE) build_satellite dates $(if $(SATELLITE_GEOJSON),--geojson "$(SATELLITE_GEOJSON)") --output-dir "$(SATELLITE_DIR)"

satellite-fetch:
	$(SATELLITE_MANAGE) build_satellite fetch $(if $(SATELLITE_GEOJSON),--geojson "$(SATELLITE_GEOJSON)") --output-dir "$(SATELLITE_DIR)"

satellite-precompute:
	$(SATELLITE_MANAGE) build_satellite precompute $(if $(SATELLITE_GEOJSON),--geojson "$(SATELLITE_GEOJSON)") --output-dir "$(SATELLITE_DIR)"

sync-satellite-dev:
	$(MAKE) sync-satellite INSTANCE=dev

sync-satellite-prod:
	$(MAKE) sync-satellite INSTANCE=prod

sync-satellite:
	@test -d "$(SATELLITE_DIR)" || { echo "error: SATELLITE_DIR='$(SATELLITE_DIR)' does not exist" >&2; exit 1; }
	@test "$(INSTANCE)" = "dev" -o "$(INSTANCE)" = "prod" || { echo "usage: make sync-satellite INSTANCE=<dev|prod>" >&2; exit 1; }
	ssh "$(REMOTE)" "sudo mkdir -p /var/lib/abies-$(INSTANCE)/data/satellite"
	rsync -rlt --delete --exclude='.worker.lock' --exclude='revisions/' --rsync-path="sudo rsync" "$(SATELLITE_DIR)/" "$(REMOTE):/var/lib/abies-$(INSTANCE)/data/satellite/"
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
