# Audit page ("Controllo")

Path: /controllo — visible to all users.

Read-only sortable-table of writes to the domain tables, sourced from
django-simple-history. Columns: time/date, user, table name,
action (insert/edit/delete), value before, value after.

Insert rows have an empty previous value and the full audited record in
the next value. Delete rows do the opposite. Update rows show the full
audited projection before and after, not only changed fields, so recovery
work has context for edits such as parcel changes. Omitted-field-only
updates are suppressed; password-only user updates are displayed as
`Password modificata.`. Materialized harvest-plan item volume
recalculations are intentionally omitted because the underlying marks and
harvest rows carry the meaningful operational changes.

## Coverage

Every model carrying `HistoricalRecords()` is audited. The authoritative
runtime check is Django's app registry: every model whose `history`
attribute is a `simple_history.manager.HistoryManager` is recorded by
django-simple-history and is expected to appear in
`apps.base.digests._audit_configs()`.

The audit generator (`apps/base/digests.py:generate_audit`) builds its
rows from `_audit_configs()` and asserts that config covers exactly the
set of history-tracked models (`_tracked_models()`); the contract is
locked by `test_audit_covers_all_tracked_models`. Adding
`HistoricalRecords()` to a model without wiring it into `_audit_configs()`
fails that test, so the log can never silently drop a tracked table. To
stop auditing a model, remove its `HistoricalRecords()` rather than
dropping it from the config.

### History-tracked models

| App/model | Base table | History table | Controllo label | Notes |
| --- | --- | --- | --- | --- |
| `base.Crew` | `base_crew` | `base_historicalcrew` | `Squadra` | Crew reference table. |
| `base.HarvestPlan` | `base_harvestplan` | `base_historicalharvestplan` | `Piano di taglio` | Plan header. |
| `base.HarvestPlanItem` | `base_harvestplanitem` | `base_historicalharvestplanitem` | `Voce piano di taglio` | Worksite/plan-item state and materialized volumes. |
| `base.HypsoParam` | `base_hypsoparam` | `base_historicalhypsoparam` | `Parametro ipsometrico` | Individual regression coefficients. |
| `base.HypsoParamSet` | `base_hypsoparamset` | `base_historicalhypsoparamset` | `Set parametri ipsometrici` | Regression set metadata and archive/supersession state. |
| `base.Parcel` | `base_parcel` | `base_historicalparcel` | `Particella` | Parcel metadata. |
| `base.SampleArea` | `base_samplearea` | `base_historicalsamplearea` | `Area di saggio` | Sample plot metadata. |
| `base.SampleGrid` | `base_samplegrid` | `base_historicalsamplegrid` | `Griglia di campionamento` | Grid metadata. |
| `base.Species` | `base_species` | `base_historicalspecies` | `Specie` | Species reference table. |
| `base.Survey` | `base_survey` | `base_historicalsurvey` | `Rilevamento` | Survey metadata. |
| `base.Tractor` | `base_tractor` | `base_historicaltractor` | `Trattore` | Tractor reference table. |
| `base.User` | `base_user` | `base_historicaluser` | `Utente` | App users and roles. |
| `ipso.IpsoUpload` | `ipso_ipsoupload` | `ipso_historicalipsoupload` | `Upload Ipso` | Staged PWA upload lifecycle. |
| `mannesi.ProductionCredit` | `mannesi_credits` | `mannesi_historicalproductioncredit` | `Acconto squadre` | Production credits/acconti. |
| `mannesi.WorkHour` | `mannesi_hours` | `mannesi_historicalworkhour` | `Ore squadre` | Team work hours. |
| `prelievi.Harvest` | `prelievi_harvest` | `prelievi_historicalharvest` | `Prelievo` | Harvest rows, including parcel and plan-item links. |

### Important non-tracked model types

These do not expose a `HistoryManager`, so they are not directly recorded
in the Controllo audit digest:

- `base.Sample`: intentionally excluded; imports create one sample per
  visited sample area, and those rows add high-volume noise. Sample
  metadata still appears through `SampleGrid`, `SampleArea`, and `Survey`.
- `base.Tree`: intentionally excluded; bulk CSV/PWA imports would swamp
  the log.
- `base.TreeSample`: intentionally excluded; these are per-tree
  measurements and are the highest-volume sampling rows.
- `base.TreeMark`: intentionally excluded; martellate CSV/PWA imports are
  bulk-ish; the linked `HarvestPlanItem` aggregate is audited instead.
- Junction/detail tables such as `prelievi.HarvestSpecies`,
  `prelievi.HarvestTractor`, `base.HypsoParamSetSurvey`, and other
  many-to-many or child rows are not individually audited unless
  represented by an audited parent or aggregate row.
- Bootstrap/reference-only tables without `HistoricalRecords()`, such as
  `Region`, `Eclass`, `Product`, and `HarvestDetail`, are not audited.
