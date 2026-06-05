# Audit page ("Controllo")

Path: /controllo — visible to all users.

Read-only sortable-table of writes to the domain tables, sourced from
django-simple-history. Columns: time/date, user, table name,
action (insert/edit/delete), value before, value after.

## Coverage

Every model carrying `HistoricalRecords()` is audited. The audit
generator (`apps/base/digests.py:generate_audit`) builds its rows from
`_audit_configs()` and asserts that config covers exactly the set of
history-tracked models (`_tracked_models()`); the contract is locked by
`test_audit_covers_all_tracked_models`. Adding `HistoricalRecords()` to a
model without wiring it into `_audit_configs()` fails that test, so the
log can never silently drop a tracked table.

`Tree`, `TreeSample` and `TreeMark` are deliberately **not**
history-tracked: they are written in bulk by CSV import (tens of
thousands of rows) and would swamp the log. To stop auditing a model,
remove its `HistoricalRecords()` rather than dropping it from the config.
