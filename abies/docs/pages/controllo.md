# Audit page ("Controllo")

Path: /abies/controllo

This page is visible to all users.

The audit page displays a sortable-table table with the following columns:

- time and date, user, table name, action (insert/edit/delete), value before, value after

This information comes from django-simple-history. The table is not editable,
but it is searchable and sortable like all other sortable-tables.
