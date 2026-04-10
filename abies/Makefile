PYTHON = python3
MANAGE = $(PYTHON) manage.py

.PHONY: check migrate import digest admin dev

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
