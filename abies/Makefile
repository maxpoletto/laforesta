PYTHON = python3
MANAGE = $(PYTHON) manage.py

.PHONY: check migrate admin dev

check:
	$(MANAGE) check

migrate:
	$(MANAGE) makemigrations
	$(MANAGE) migrate

admin:
	$(MANAGE) createsuperuser

dev: migrate admin
