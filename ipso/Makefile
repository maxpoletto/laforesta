.PHONY: reference icons test serve deploy clean

# Output location on the laforesta.it VM. The user is provisioning the
# subdomain (DNS + apache vhost + TLS) via ../system/ansible separately.
DEPLOY_HOST    ?= ipso.laforesta.it
DEPLOY_DIR     ?= /var/www/ipso/html

# Source files that ship with the deploy. tools/, tests.js, and Makefile
# are not shipped.
DEPLOY_EXCLUDES = \
	--exclude=.git \
	--exclude=tools \
	--exclude=tests.js \
	--exclude=Makefile \
	--exclude=README.md \
	--exclude=plans

reference:
	python3 tools/build_reference.py

icons:
	python3 tools/build_icons.py

test:
	node tests.js

# Local HTTPS preview using `python3 -m http.server` over plain HTTP.
# Service worker registration requires HTTPS or localhost; localhost
# counts as secure for SW purposes.
serve:
	@echo "Serving at http://localhost:8000/ — open in a browser."
	@python3 -m http.server 8000

deploy: reference icons test
	rsync -avz --delete $(DEPLOY_EXCLUDES) ./ $(DEPLOY_HOST):$(DEPLOY_DIR)/

clean:
	rm -f reference.json icons/*.gif icons/*.png
