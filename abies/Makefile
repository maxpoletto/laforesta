PYTHON = python3
MANAGE = $(PYTHON) manage.py

.PHONY: check migrate import admin dev

check:
	$(MANAGE) check

migrate:
	$(MANAGE) makemigrations
	$(MANAGE) migrate

import:
	$(PYTHON) ingest/import_reference.py
	$(PYTHON) ingest/import_parcels.py
	$(PYTHON) ingest/import_mannesi.py

admin:
	$(MANAGE) createsuperuser

dev: migrate import admin
