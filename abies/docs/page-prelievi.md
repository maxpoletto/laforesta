# Prelievi page

This page supports recording and display of harvesting operations.

## Visual appearance

A top filter bar hosts a double-ended year slider (`Anni`, see
`base/.../range-slider.js`), search box (`Filtra`), "Azzera filtri" reset, and
"Esporta CSV" export. Below it sit three collapsible sections, separated by the
standard dark-green 4px rule:

1. **Produzione** — stacked bar chart of total quintals over time, with a
   per-chart pull-down selector for the breakdown dimension (Totale /
   Compresa / Particella / Squadra / Specie / Trattore / Tipo) and a
   "mesi" checkbox that toggles between year-granularity and
   month-granularity buckets. When the category count exceeds 12, the
   tail is collapsed into an "Altro" series.
2. **Specie per particella** — stacked bar chart with `<compresa>/<particella>`
   on the x-axis and one species stack per bar, sorted by total.
3. **Interventi** — harvest-operations in a sortable-table, as in UI Design
   Patterns > Tabular Data.

Sections 1 and 2 are collapsed by default; section 3 is open. Chart
sections render lazily (only when first opened) and re-render whenever
active filter set changes.

The full dataset is served as a single compressed JSON digest. All filtering (by
year and search box) is client-side and affects both charts and table.

Table columns are:
Data, Compresa, Particella, Squadra, VDP, Tipo, Q.li, Volume (m³), Note,
Altre note, (quintal columns by **major** species in sort_order), (quintal
columns by tractor in alphabetical order). See "Minor species and Altro" below.

All quintal values display with one decimal and comma separator (Italian locale,
e.g., "164,0"). Per-species and per-tractor quintal columns show blank for zero.
VDP displays as an integer. Columns have fixed widths; the table scrolls
horizontally when the viewport is too narrow.

### Add/edit form

The add/edit form is laid out as a compact grid (three fields per row):

- Row 1: date
- Row 2: "Cantiere" pulldown, "Particella" pulldown.
- Row 3: "Squadra" pulldown, "Tipo" (product) pulldown, "Q.li" (step 0.1).
- Row 4: "VDP" numeric, "Altre note" text.

**Cantiere pulldown.** Options are `harvest_plan_item` rows with `state ∈ {open,
harvesting}`, rendered as `<Plan> · <Year> · <Region> <Parcel>` (or
`<Region> (tutti)` for region-wide items). Required for new harvests.

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
represented by the  "Altro" entry. Each row has a numeric input and a "100%"
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
    tokens identifying which sections are expanded. Tokens: `a` =
    Produzione chart, `b` = Specie-per-particella chart, `i` = Interventi
    (the table itself). Absent means the default (`i` only). An explicit
    empty value (`?o=`) means all sections collapsed.
  - Production chart breakdown: `b=total|compresa|particella|squadra|specie|trattore|tipo`
    (absent = `total`).
  - Production chart monthly granularity: `m=1` (absent = year granularity).

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
| Harvest save (create or update) | `prelievi`, `audit`; + `harvest_plan_items` if linked to a plan item | `prelievi` and linked `harvest_plan_items` via `patches` |
| Harvest delete | same as save | `prelievi` via `deletes`; `harvest_plan_items` via `patches` |
| Species save (Settings → Trees) | `prelievi`, `species`, `audit` | — |
| Tractor save (Settings) | `prelievi`, `audit` | — |
| Crew save (Settings) | `prelievi`, `audit` | — |

The last three are **cross-domain**: the writes live in the `impostazioni`
domain but the prelievi digest reads from `Species` (its `minor` flag,
`sort_order`, and `common_name` define the species column set), `Tractor`
(`manufacturer`/`model` are the tractor column labels), and `Crew` / `Product`
(name value columns). A settings edit that changes any of these must mark
`prelievi` stale, or the column set / labels go silently stale until the next
harvest write happens to mark it (see `data-architecture.md` §"JSON digest
regeneration"). Products are not editable in Settings, so they need no mark.

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

## Minor species and Altro

Species with `Species.minor = True` are uncommon and are grouped under a single
"Altro" entry to keep the input form compact and the table narrow. The minor
flag is prelievi-specific: minor species remain fully available as individual
entries in Campionamenti (sampled trees), Piani di taglio (tree marks), and
Settings.

The set is controlled by the `minor` field on each `Species` record (seeded from
`apps/base/data/species.csv`, editable via Settings → Trees).

The "Altro" species itself (`common_name == S.SPECIES_OTHER`) is the bucket the
minor species fold into, so it must always stay major. The species save view
rejects flagging it minor (`ERR_OTHER_NOT_MINOR`) and the Settings form disables
its Minore checkbox; `prelievi_species_cols()` relies on exactly one major
species named "Altro".

**Input form:** only (non-minor, active) species appear. When editing a legacy
harvest that used a minor species directly, its percentage is folded into the
Altro row so the form percentages still sum to 100.

**Digest / table / charts:** the prelievi digest emits one column per
major species only. Quintals and percentages for minor species are
aggregated into the Altro column. This is handled by
`prelievi_species_cols()` and `aggregate_sp_pcts()` in
`apps/base/digests.py`, shared by `generate_prelievi()`,
`build_harvest_record()`, and the form context builder.

**Volume calculation:** new harvests entered with Altro use the Altro species
density (9.00 q/m³, a rough average). (OK because minor species are a tiny
fraction of total volume.) Legacy harvests recorded with individual minor
species retain their original volume; re-saving them through the form rewrites
the junction rows as Altro, recomputing volume with the Altro density.
