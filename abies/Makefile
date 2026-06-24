PYTHON = python3
CANONICAL_DIR ?= data/canonical
BRANDING_DIR ?= data/branding
DJANGO_DEBUG ?= 1
DJANGO_SECRET_KEY ?= django-insecure-local-dev
LANGUAGE_SUFFIX ?= _it
STATIC_ROOT ?= staticfiles

REMOTE_USER ?= maxp
REMOTE_HOST ?= abies.laforesta.it
REMOTE ?= $(REMOTE_USER)@$(REMOTE_HOST)
REMOTE_ABIES_USER ?= abies
REMOTE_ABIES_GROUP ?= abies

ref ?=
REF ?= $(ref)
DEPLOY_ARGS ?=

MANAGE = env DJANGO_DEBUG="$(DJANGO_DEBUG)" DJANGO_SECRET_KEY="$(DJANGO_SECRET_KEY)" $(PYTHON) manage.py

# Active language. Parallel to `from config.strings_it import *` in
# config/strings.py and `export * from './strings_it.js'` in strings.js:
# template files are committed as <name><LANGUAGE_SUFFIX>.html; the
# unsuffixed <name>.html symlinks are materialized at install time.

.PHONY: help \
        check migrate reset-db load-canonical-data bootstrap geo stage-marks-uploads digest admin link-templates sync-branding dev serve \
        collectstatic minify \
        deploy-dev deploy-prod stage-canonical stage-canonical-dev stage-canonical-prod bootstrap-dev bootstrap-prod docker-prune \
        test test-js \
        clean reset-schema update-vendor

help:
	@echo "Local development:"
	@echo "  make dev                         reset local DB, load canonical data, create admin"
	@echo "  make serve                       run Django dev server in the background"
	@echo "  make test                        run JS tests and pytest"
	@echo "  make sync-branding                copy data/branding assets into static files"
	@echo "  make collectstatic minify        collect static files, then minify collected JS"
	@echo ""
	@echo "Remote deployment:"
	@echo "  make deploy-dev [REF=<ref>]      deploy current tree or ref to abies-dev"
	@echo "  make deploy-prod REF=<ref>       deploy clean tagged/ref build to production"
	@echo "  make stage-canonical-dev         rsync CANONICAL_DIR to dev data/canonical"
	@echo "  make stage-canonical-prod        rsync CANONICAL_DIR to prod data/canonical"
	@echo "  make bootstrap-dev               stage canonical data, then bootstrap dev"
	@echo "  make bootstrap-prod              stage canonical data, then bootstrap prod"
	@echo ""
	@echo "Useful variables: CANONICAL_DIR=$(CANONICAL_DIR), BRANDING_DIR=$(BRANDING_DIR), REMOTE=$(REMOTE), DEPLOY_ARGS=--reset-user"

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
	env DJANGO_DEBUG="$(DJANGO_DEBUG)" DJANGO_SECRET_KEY="$(DJANGO_SECRET_KEY)" PYTHON="$(PYTHON)" bin/load-canonical-data $(CANONICAL_DIR)

bootstrap:
	$(MANAGE) bootstrap $(CANONICAL_DIR)

geo:
	mkdir -p data/geo
	$(MANAGE) build_geo $(CANONICAL_DIR)

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
	./bin/deploy dev $(REF) $(DEPLOY_ARGS)

deploy-prod:
	@test -n "$(REF)" || { echo "usage: make deploy-prod REF=<tag-or-sha> [DEPLOY_ARGS=--reset-user]" >&2; exit 1; }
	./bin/deploy prod $(REF) $(DEPLOY_ARGS)

stage-canonical-dev:
	$(MAKE) stage-canonical INSTANCE=dev

stage-canonical-prod:
	$(MAKE) stage-canonical INSTANCE=prod

stage-canonical:
	@test -d "$(CANONICAL_DIR)" || { echo "error: CANONICAL_DIR='$(CANONICAL_DIR)' does not exist" >&2; exit 1; }
	@test "$(INSTANCE)" = "dev" -o "$(INSTANCE)" = "prod" || { echo "usage: make stage-canonical INSTANCE=<dev|prod>" >&2; exit 1; }
	ssh "$(REMOTE)" "sudo mkdir -p /var/lib/abies-$(INSTANCE)/data/canonical"
	rsync -rlt --delete --rsync-path="sudo rsync" "$(CANONICAL_DIR)/" "$(REMOTE):/var/lib/abies-$(INSTANCE)/data/canonical/"
	ssh "$(REMOTE)" "sudo chown -R $(REMOTE_ABIES_USER):$(REMOTE_ABIES_GROUP) /var/lib/abies-$(INSTANCE)/data/canonical"
	ssh "$(REMOTE)" "sudo chmod -R u=rwX,g=rX,o= /var/lib/abies-$(INSTANCE)/data/canonical"

bootstrap-dev: stage-canonical-dev
	./bin/bootstrap-data dev

bootstrap-prod: stage-canonical-prod
	./bin/bootstrap-data prod

docker-prune:
	docker --context vm-abies image prune

# --- Tests ------------------------------------------------------------------

test: test-js
	env DJANGO_DEBUG="$(DJANGO_DEBUG)" DJANGO_SECRET_KEY="$(DJANGO_SECRET_KEY)" $(PYTHON) -m pytest --cov --cov-report=term-missing -q

# Pure-JS tests for shared client-side modules.
test-js:
	node apps/ipso/static/ipso/pwa/session.test.mjs
	node apps/ipso/static/ipso/pwa/csv.test.mjs
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
