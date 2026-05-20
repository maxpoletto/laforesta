# Prelievi page

The prelievi page supports recording and display of harvesting operations.

- Path: /prelievi
- Query parameters:
  - Year range: `y1=YYYY`, `y2=YYYY` (date slider bounds)
  - Region: `c=N` — id of the region (compresa).  Restricts the
    displayed set to that region.  Stale / unknown id falls back to
    the unscoped view.
  - Particella: `pa=N` — id of the parcel.  Restricts further to one
    parcel.  `parcel.id` already disambiguates region (parcel names
    are only unique within a region; the id is globally unique), so
    `pa=` does not strictly need `c=`.  However, the standard
    cross-page link pattern emits both for symmetry and to keep the
    region pulldown in the resulting view set correctly.
  - Sort column: `sc=N`
  - Sort order: `so=0/1` (ascending/descending)
  - Filter: `f=...` (URL-encoded sortable-table search string,
    applied on top of the `c=` / `pa=` scope).
  - Open collapsible sections: `o=...`, a concatenation of single-char
    tokens identifying which sections are expanded.  Tokens: `a` =
    Produzione chart, `b` = Specie-per-particella chart, `i` = Interventi
    (the table itself).  Absent means the default (`i` only).  An explicit
    empty value (`?o=`) means all sections collapsed.
  - Production chart breakdown: `b=total|compresa|particella|squadra|specie|trattore|tipo`
    (absent = `total`).
  - Production chart monthly granularity: `m=1` (absent = year granularity).

`c=` and `pa=` exist primarily to support unambiguous cross-page
links from Piano di taglio (status-chip click) and Bosco (per-parcel
"Produzione storica" → Prelievi).  They behave as a hard scope: rows
outside the (region, particella) are excluded from both the table
and the chart sections, just as if the user had typed an exact
match into the search box, but without the false-positive risk of
substring matches on a homonymous particella in another region.

## Visual appearance

A top filter bar hosts the year slider (`Anni`), search box (`Filtra`),
"Azzera filtri" reset, and "Esporta CSV" export.  Below it sit three
collapsible sections, separated by the standard dark-green 4px rule:

1. **Produzione** — stacked bar chart of total quintals over time, with a
   per-chart pull-down selector for the breakdown dimension (Totale /
   Compresa / Particella / Squadra / Specie / Trattore / Tipo) and a
   "mesi" checkbox that toggles between year-granularity and
   month-granularity buckets.  When the category count exceeds 12, the
   tail is collapsed into an "Altro" series.
2. **Specie per particella** — stacked bar chart with `<compresa>/<particella>`
   on the x-axis and one species stack per bar, sorted by total.
3. **Interventi** — the full harvest-operations sortable-table described
   in "Tabular data" in CLAUDE.md.

Sections 1 and 2 are collapsed by default; section 3 is open.  Chart
sections render lazily (only when first opened) and re-render whenever
the active filter set changes.

The full dataset is served as a single compressed JSON digest.  All
filtering is client-side: a double-ended date slider (see
`bosco/a/range-slider.*`) with year granularity restricts the displayed
range, and the search box filters within that range.  Both charts and the
table react to the same filter set.  No server round-trips for filtering.

Table columns are:
Data, Compresa, Particella, Squadra, VDP, Tipo, Q.li, Note, Altre note,
(quintal columns by species in sort_order), (quintal columns by tractor
in alphabetical order).

All quintal values display with one decimal and comma separator (Italian locale,
e.g., "164,0"). Per-species and per-tractor quintal columns show blank for zero.
VDP displays as a plain integer (no thousands separator). Columns have fixed
widths; the table scrolls horizontally when the viewport is too narrow.

The add/edit form is laid out as a compact grid (three fields per row):

- Row 1: date picker, "Cantiere" pull-down. The Cantiere options are
  the `harvest_plan_item` rows whose `state ∈ {open, harvesting}`,
  rendered as `<Compresa>/<Particella>` (or `<Compresa>/(tutti)` for
  region-wide items). Selection is **required** for new harvests; this
  is the only path to creating a harvest in v1 (every harvest must
  tie back to an approved intervento — there is no escape hatch).
- Row 2: "Squadra" pull-down, "Tipo" (product) pull-down, "Q.li"
  numeric input (step 0.1).
- Row 3: "VDP" numeric input, "Altre note" text input.

Below Row 3, a read-only summary line displays the `damaged`,
`unhealthy`, and `psr` flags of the selected Cantiere — rendered as a
comma-joined string from `"Catastrofato"`, `"Fitosanitario"`, `"PSR"`
— and they are auto-applied to the new harvest on save. This matches
the schema trigger
`harvest.{damaged,unhealthy,psr} == harvest_plan_item.{...}` so the
user never sets them directly on the harvest form.

Prot (record2) is not shown in the form — it is display-only for legacy data.
Existing Prot values are preserved on edit; new records never have a Prot value.

VDP must be unique (checked server-side against all records).

Species and tractor percentage sections appear side-by-side below the main
fields. For each, all active choices are listed with a numeric input and a
"100%" quick-set button. Species and tractor percentages must each sum to 100
(validated both client-side and server-side). Pressing a "100%" button sets
that input to 100 and the others in the same group to 0.

Additional validation (client-side and server-side):
- Date cannot be in the future.
- Q.li must be positive.

On validation error for a new entry, the form re-populates with the submitted
values so the user does not lose their input.

The form has two submit buttons: "Salva" (save and return to the table view)
and "Salva e aggiungi" (save and present a blank form for the next entry).
"Salva e aggiungi" supports the common batch-entry workflow where office staff
enter a stack of paper slips in sequence.

## Data tables

- parcels.json: as above
- crews.json: JSON version of the crew
- prelievi.json: a de-normalized version of the harvest table, containing
  everything in the sortable-table as well as percentage values (to support
  prepopulating the edit form).  Includes a materialized `Volume (m³)`
  column from `harvest.volume_m3` (see `database.md`); displayed in the
  table as a small companion to `Q.li`.
- parcel_year_production.json: per-(region, parcel, year) totals.  Columns:
  `Compresa`, `Particella`, `Anno`, `Q.li`, `Volume (m³)` — both unit sums
  are materialized so the Piano di taglio calendar can compute its
  status chip in m³ without touching prelievi.json.

Successful writes on the backend mark the prelievi.json and
parcel_year_production.json digests as stale.

`harvest.volume_m3` is itself materialized at write time using the
species densities current at that moment.  Editing a species' density
later does *not* retroactively recompute existing harvests — same
capture-at-write-time pattern used elsewhere — so the digests remain
valid.  Only newly written harvests pick up the new density value.
