# Prelievi page

This page supports recording and display of harvesting operations.

## Visual appearance

A top filter bar hosts a double-ended year slider (`Anni`, see
`base/.../range-slider.js`), search box (`Filtra`), "Azzera filtri" reset, and
"Esporta" export. Below it sit three collapsible sections, separated by the
standard dark-green 4px rule:

1. **Riassunto** — two stacked bar charts:
   - **Per Anno**, with total quintals over time. It has a pull-down selector
     for the breakdown dimension: (Totale), Compresa, Particella, Squadra,
     Specie, Trattore, Tipo. It also has a "Per mese" checkbox that toggles between
     year-granularity and month-granularity buckets.
   - **Per Particella**, with `<compresa>/<particella>` on the x-axis, sorted
     by total. It has a pull-down selector for (Totale), Squadra, Specie,
     Trattore, Tipo.
   When a chart category count exceeds 12, the tail is collapsed into an
   "Altro" series.
2. **Calendario** — a CDT-style grid with parcels on the y-axis, grouped by
   compresa, and harvest years (`YYYY`) on the x-axis by default. A "Per mese"
   checkbox switches the x-axis to harvest months (`YYYY-MM`). A dark-green
   cell marks every parcel/period combination with at least one harvest row in
   the currently filtered dataset. Clicking a populated cell writes a column
   filter such as `data:2019 compresa:Fabrizia particella:14b` or
   `data:2019-06 compresa:Fabrizia particella:14b` into the search box,
   further restricting the charts, calendar, and table to that period and
   parcel while preserving unrelated search terms such as `squadra:zaffino`.
3. **Interventi** — harvest-operations in a sortable-table, as in UI Design
   Patterns > Tabular Data.

Sections 1 and 2 are collapsed by default; section 3 is open. Summary charts
and the calendar render lazily (only when first opened) and re-render whenever
the active filter set changes.

The full dataset is served as a single compressed JSON digest. All filtering (by
year and search box) is client-side and affects the charts, calendar, and table.

Qualified parcel filters (`part:1` or `particella:1`) match the entire parcel
identifier: only `1`, excluding `1b`, `10`, `10b`, etc. Matching is
case-insensitive and preserves distinct identifiers such as `01`. This applies
to calendar-generated filters as well as typed ones, and therefore to charts
and exports too. Sorting the parcel column uses natural order:
`1`, `1b`, `2`, `2a`, …, `10`, `11`; descending reverses that order.

"Esporta" submits the row IDs retained by those filters and downloads the
normalized harvest data in the same schema accepted by "Importa CSV". The
backup keeps the three flags separate, retains every individual species
(including minor species), uses percentage columns for species and tractors,
and includes the materialized capture-time volume so later density edits do not
rewrite history during restore. An optional `ID cantiere` column reconnects
Abies-originated backups to their existing cutting-plan items; older external
CSVs without that column retain the legacy unlinked-import behavior.

Table columns are:
Data, Compresa, Particella, Squadra, VDP, Tipo, Q.li, Volume (m³), Note,
Altre note, (quintal columns by **major** species in sort_order), (quintal
columns by tractor in alphabetical order). See "Minor species and catch-alls" below.

All quintal values display with one decimal and comma separator (Italian locale,
e.g., "164,0"). Per-species and per-tractor quintal columns show blank for zero.
VDP displays as an integer. Columns have fixed widths; the table scrolls
horizontally when the viewport is too narrow.

A column-aligned `Totali` footer sums `Q.li`, `Volume (m³)`, every displayed
major-species column, and every tractor column across the complete current
filter result, not just the visible pagination page. Values use the same decimal
precision as their columns but omit unit suffixes, since the units are already
in the headers. VDP is an identifier and is not summed.

### Add/edit form

The add/edit form is laid out as a compact grid (three fields per row):

- Row 1: date
- Row 2: "Cantiere" pulldown, "Particella" pulldown.
- Row 3: "Squadra" pulldown, "Tipo" (product) pulldown, "Q.li" (step 0.1).
- Row 4: "VDP" numeric, "Altre note" text.

**Cantiere pulldown.** Options are `harvest_plan_item` rows with `state ∈ {open,
harvesting}`, rendered as
`<Year> · <Region> <Parcel> · Approvato il <DD/MM/YYYY>` (or
`<Region> (tutti)` for region-wide items). The approval date is the earliest
opening (`open = true`) in the item's `harvest_transition` history, retained
when a closed Cantiere is reopened. The selected option in an edit form uses
the same label, including for closed Cantieri. Required for new harvests.

- Parcel-scoped items: Particella pulldown hidden; parcel derived
  from the item.
- Region-wide items (`parcel IS NULL`): Particella pulldown visible
  and required (filtered to the item's region). Server validates the
  match.

Below row 2 (Cantiere pulldown), a read-only line shows the selected
Cantiere's `damaged`/`unhealthy`/`psr` flags (auto-applied to the harvest
on save; the schema trigger enforces consistency).

Legacy imports (`harvest_plan_item_id IS NULL`) remain editable;
editing without touching Cantiere preserves the NULL link.

Prot (record2) is not shown in the form; it is backend-only for legacy data.

VDP must be unique (checked server-side against all records) and defaults to
max(existing VDP)+1.

Species and tractor percentage sections appear side-by-side below the main
fields. Only non-minor (see below) species are listed; minor species are
represented by `Altra conifera` or `Altra latifoglia` according to botanical type.
Legacy `Altro` is absent from new forms but retained when editing a harvest
that already used it. Each row has a numeric input and a "100%"
quick-set button that also zeros other rows. Species and tractor percentages
must each sum to 100 (validated on client and server).

Additional validation (client and server):
- Date cannot be in the future.
- Q.li must be positive.

Validation errors appear directly in the form (UI Design Patterns > Error
reporting).

The form has [Annulla/Salva/Salva e continua] buttons. See UI Design Patterns >
Bottom-of-form button layout.

## URL parameters

- Path: /prelievi
- Query parameters:
  - Year range: `y1=YYYY`, `y2=YYYY` (date slider bounds)
  - Optional forest scope: `c=<region_id>`, `pa=<parcel_id>`. Used by
    Bosco links to show harvests for a region or one parcel.
  - Sort column: `sc=N`
  - Sort order: `so=0/1` (ascending/descending)
  - Filter: `f=...` (URL-encoded sortable-table search string).
  - Open collapsible sections: `o=...`, a concatenation of single-char
    tokens identifying which sections are expanded. Tokens: `a` = Riassunto,
    `b` = Calendario, `i` = Interventi (the table itself). Absent means the
    default (`i` only). An explicit empty value (`?o=`) means all sections
    collapsed.
  - Per Anno chart breakdown: `b=total|compresa|particella|squadra|specie|trattore|tipo`
    (absent = `total`).
  - Per Particella chart breakdown: `pb=total|squadra|specie|trattore|tipo`
    (absent = `total`).
  - Per Anno chart monthly granularity: `m=1` (absent = year granularity).
  - Calendar monthly granularity: `cm=1` (absent = year granularity).

## Data tables

- parcels.json: as above
- crews.json: JSON version of the crew
- prelievi.json: a de-normalized version of the harvest table, containing
  everything in the sortable-table as well as percentage values (to support
  prepopulating the edit form). Hidden `Region id` and `Parcel id` columns
  support stable cross-page filters. Includes a materialized `Volume (m³)`
  column from `harvest.volume_m3` (see `database.md`); displayed in the
  table as a small companion to `Q.li`.

### Cache invalidation

| Write | Digests marked stale | Optimistic client patch |
|---|---|---|
| Harvest save (create or update) | `prelievi`, `audit`; + `harvest_plan_items`, `future_production` if linked to a plan item | `prelievi` and linked `harvest_plan_items` via `patches` |
| Harvest delete | same as save | `prelievi` via `deletes`; `harvest_plan_items` via `patches` |
| Species save (Impostazioni → Specie) | `prelievi`, `species`, Bosco species digests, every existing `sampled_trees_*` / `mark_trees_*`, `audit` | `species` row patched; all dependent full/dynamic caches evicted |
| Tractor save (Impostazioni → Trattori) | `prelievi`, `audit` | — |
| Crew save (Squadre → Personale) | `prelievi`, `audit` | — |

The last three are **cross-domain**: the writes live outside Prelievi,
while the prelievi digest reads from `Species` (its `minor` flag, `sort_order`,
and `common_name` define the species column set), `Tractor`
(`manufacturer`/`model` are the tractor column labels), and `Crew` / `Product`
(name value columns). Edits that change any of these must mark `prelievi` stale,
or the column set / labels go silently stale until the next harvest write happens
to mark it (see `data-architecture.md` §"JSON digest regeneration"). Products
are not editable in these pages, so they need no mark.

Harvest-op deletion cascades to its `harvest_species` and `harvest_tractor`
junction rows.

When a harvest is linked to a `harvest_plan_item`, the save view also
re-materializes `volume_actual_m3` on the item (aggregate sum of all
linked harvests). The first linked harvest insert auto-advances the
item state from `open` to `harvesting`.

`harvest.volume_m3` is itself materialized at write time using the
species densities current at that moment. Editing a species' density
later does *not* retroactively recompute existing harvests — same
capture-at-write-time pattern used elsewhere — so the digests remain
valid. Only newly written harvests pick up the new density value.

## Minor species and catch-alls

Species with `Species.minor = True` are uncommon and are grouped under
`Altra conifera` or `Altra latifoglia` to keep the input form compact and the
table narrow. `Species.botanical_type` determines the target. The minor
flag is prelievi-specific: minor species remain fully available as individual
entries in Campionamenti (sampled trees), Piani di taglio (tree marks), and
Impostazioni.

The set is controlled by the `minor` field on each `Species` record (seeded from
`apps/base/data/species.csv`, editable via Impostazioni → Specie).

All three catch-all rows must remain major and retain their reserved names.
Settings protects these invariants and fixes the botanical types of the two
new rows. `Altro` remains unclassified; its existing junctions and tree foreign
keys are never automatically relabeled as conifers or broadleaves. Unrecognized
custom minor taxa also retain the legacy grouping until their classification
is explicitly set in Settings.

**Input form:** new forms offer active, major species except Altro, including
the two new catch-alls. Editing preserves existing legacy/inactive species
percentages. When a saved harvest uses individual minor species, the form folds
their percentages into the corresponding botanical buckets. Direct new-entry
POSTs cannot bypass the exclusion of Altro or submit arbitrary/minor species.

**Digest / table / charts:** the prelievi digest emits one column per
major species only. Quintals and percentages for minor species are
aggregated into their botanical-type columns, while historical Altro remains
a separate column. This is handled by
`prelievi_species_cols()` and `aggregate_sp_pcts()` in
`apps/base/digests.py`, shared by `generate_prelievi()`,
`build_harvest_record()`, and the form context builder.

**Volume calculation:** new harvests use the density of each submitted species,
including the new catch-alls. Their initial defaults match Abete bianco and
Faggio; subsequent density edits are independent. Existing harvests retain
their captured volume, and CSV exports preserve their original individual
species. Re-saving a harvest through the percentage form folds individual
minor junctions into their botanical buckets and recomputes volume using those
buckets' current densities, as the existing minor-species edit flow did with
Altro.
