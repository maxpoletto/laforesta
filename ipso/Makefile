.PHONY: build test serve local-test deploy deploy-test icons clean

# Output location on the laforesta.it VM. The user is provisioning the
# subdomain (DNS + apache vhost + TLS) via ../system/ansible separately.
DEPLOY_HOST ?= ipso.laforesta.it
DEPLOY_DIR  ?= /var/www/ipso/html

DATA_DIR    ?= ../abies-data

# Stage src/ + generated artifacts into build/. Deploy targets rsync
# from build/ — the source tree is never touched. Generated artifacts
# (reference.json, geo.js, terreni.geojson) live in build/ only.
build:
	@if [ ! -f secrets/upload_config.json ]; then \
	  echo "ERROR: secrets/upload_config.json missing."; \
	  echo "  Copy secrets/upload_config.json.example and fill in the token."; \
	  exit 1; \
	fi
	mkdir -p build
	rsync -a --delete src/ build/
	python3 tools/build_reference.py     --data-dir $(DATA_DIR) build/reference.json
	python3 tools/vendor_geo.py          build/geo.js
	python3 tools/build_upload_config.py secrets/upload_config.json build/upload-config.js
	cp $(DATA_DIR)/terreni.geojson       build/terreni.geojson

test: build
	node test/tests.js
	$(MAKE) -C upload-server test

# Local HTTPS preview. Service worker registration requires HTTPS or
# localhost; localhost counts as secure for SW purposes. Pure static —
# no /upload backend. Use `make local-test` for the upload-server flow.
serve: build
	@echo "Serving build/ at http://localhost:8001/ — open in a browser."
	@cd build && python3 -m http.server 8001

# Full local stack: build, start ipso-upload on 127.0.0.1:8765, run a
# small reverse proxy on http://127.0.0.1:8000/ that serves build/ and
# proxies /upload to the upload-server. Mimics production topology so
# the browser doesn't see a cross-origin request. Ctrl-C to stop both.
local-test: build
	@bin/local-test

deploy: build test
	rsync -avz --delete build/ $(DEPLOY_HOST):$(DEPLOY_DIR)/
	@$(MAKE) -C upload-server deploy DEPLOY_HOST=$(DEPLOY_HOST)

# Deploy with test polygons and a matching synthetic reference.json so
# the test compresa appears in the pre-session pulldown. Same target
# host as `deploy`; bump version.js or clear site data on the phone
# when flipping between variants so the SW refetches.
deploy-test: build test
	cp test/test.geojson                  build/terreni.geojson
	python3 tools/build_test_reference.py build/reference.json
	rsync -avz --delete build/ $(DEPLOY_HOST):$(DEPLOY_DIR)/

# Rebuild the committed icons from the source logo. Writes to
# src/img/ — rerun `make build` to pick them up. Not invoked by build.
icons:
	python3 tools/build_icons.py

clean:
	rm -rf build
	$(MAKE) -C upload-server clean
