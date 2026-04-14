# Prelievi page

The prelievi page supports recording and display of harvesting operations.

- Path: /abies/prelievi
- Query parameters:
  - Year range: `y1=YYYY`, `y2=YYYY` (date slider bounds)
  - Sort column: `sc=N`
  - Sort order: `so=0/1` (ascending/descending)
  - Filter: `f=...` (URL-encoded sortable-table search string)
  - Open collapsible sections: `o=...`, a concatenation of single-char
    tokens identifying which sections are expanded.  Tokens: `a` =
    Produzione chart, `b` = Specie-per-particella chart, `i` = Interventi
    (the table itself).  Absent means the default (`i` only).  An explicit
    empty value (`?o=`) means all sections collapsed.
  - Production chart breakdown: `b=total|compresa|particella|squadra|specie|trattore|tipo`
    (absent = `total`).
  - Production chart monthly granularity: `m=1` (absent = year granularity).

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

- Row 1: date picker, "Compresa" pull-down (cascades to filter Particella),
  "Particella" pull-down.
- Row 2: "Squadra" pull-down, "Tipo" (optype) pull-down, "Q.li" numeric input
  (step 0.1).
- Row 3: "VDP" numeric input, "Note" pull-down, "Altre note" text input.

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
  prepopulating the edit form).

Successful writes on the backend mark the prelievi.json and
parcel_year_production.json digests as stale.
