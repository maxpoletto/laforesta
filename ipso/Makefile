.PHONY: build test serve deploy deploy-test icons clean

# Output location on the laforesta.it VM. The user is provisioning the
# subdomain (DNS + apache vhost + TLS) via ../system/ansible separately.
DEPLOY_HOST ?= ipso.laforesta.it
DEPLOY_DIR  ?= /var/www/ipso/html

BOSCO_DATA   = ../bosco/data

# Stage src/ + generated artifacts into build/. Deploy targets rsync
# from build/ — the source tree is never touched. Generated artifacts
# (reference.json, geo.js, terreni.geojson) live in build/ only.
build:
	mkdir -p build
	rsync -a --delete src/ build/
	python3 tools/build_reference.py build/reference.json
	python3 tools/vendor_geo.py       build/geo.js
	cp $(BOSCO_DATA)/terreni.geojson  build/terreni.geojson

test: build
	node test/tests.js

# Local HTTPS preview. Service worker registration requires HTTPS or
# localhost; localhost counts as secure for SW purposes.
serve: build
	@echo "Serving build/ at http://localhost:8000/ — open in a browser."
	@cd build && python3 -m http.server 8000

deploy: build test
	rsync -avz --delete build/ $(DEPLOY_HOST):$(DEPLOY_DIR)/

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
