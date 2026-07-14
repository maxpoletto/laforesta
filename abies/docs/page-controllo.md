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

`Sample`, `Tree`, `TreeSample`, `TreeMark`, and `TreePreserved` are deliberately
**not** history-tracked: they are written by bulk-ish CSV import paths
(tens of thousands of rows for sampled/marked trees) and would swamp the
log. Their parent or aggregate rows carry the user-visible event where
practical; sample metadata still appears through `SampleGrid`, `SampleArea`,
and `Survey`. To stop auditing a model, remove its `HistoricalRecords()`
rather than dropping it from the config.

Known intentional non-history tables include reference tables loaded only
by bootstrap and child/junction tables whose parent row is the audited
user-facing object, such as harvest species/tractor percentages.
