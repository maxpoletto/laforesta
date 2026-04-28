# Campionamenti page

## Overview

Recording and exploration of forest sampling operations.  A *sampling*
takes place in a circular *sample area* (typically ~12 m radius) where
every tree inside the radius is measured: diameter at breast height,
total height, outer-10-rings radial width, and (for coppice) shoot
counter within the stump.  Per-hectare biomass and stand-structure
statistics are extrapolated from these measurements to the parcel and
region.

A *sessione* (TBD name — see "Open questions") is a logical grouping
of sample-area visits performed over a contiguous period, typically a
season or a year, and usually tied to a specific harvest plan.  A
session is not a separate database table; it is derived from
`sample.date` (year) plus `sample.harvest_plan_id`.  Within a session,
each sample area is visited at most once.

This page is the canonical surface for entering new sampling data and
for browsing existing sampling history.

## Visual layout

A top filter bar (same idiom as Prelievi) hosts:

- "Anni" double-ended year slider.
- "Filtra" search box.
- "Azzera filtri" reset button.
- "Esporta CSV" export.

Below the filter bar sit three collapsible sections separated by
dark-green 4 px horizontal rules:

1. **Sessioni** — high-level table of sampling sessions.

   Columns: anno, piano di taglio, n. rilievi, n. alberi, particelle
   coperte (count), note.  One row per (anno, piano) combination
   present in the sample table.

   Selecting a row (radio-style; click the row to select) filters
   sections 2 and 3 to that session.  Clicking the selected row again
   deselects, returning to the all-sessions view.

   Writers see a pencil icon per row that opens a notes-editor for
   the session label/notes (stored in… TBD — possibly a small
   `sampling_session` table keyed on (year, plan_id), or a new
   `notes` field on `sample`).

2. **Mappa** — Leaflet map covering the current region (region pulldown
   in the standard map nav bar; see "Maps" in `CLAUDE.md`).  Shows
   parcel borders and sample-area circles drawn at their actual
   `r_m` radius.

   When no session is selected: all sample areas of the region are
   tinted a single color.  When a session is selected: only that
   session's sample areas are colored; other (historical) sample
   areas show as faint outlines for context.

   Click a sample area to "drill in": section 3 below filters to the
   trees of that sample area.  Click again to deselect.

   Writers see a "+ Nuovo rilievo" button at the bottom of the map
   panel which opens the new-rilievo form (see below).

3. **Alberi campionati** — tree-level sortable-table.

   Default scope: all sampled trees in the region (and within the
   year-slider range).  When a session is selected in section 1, the
   scope narrows to that session.  When a sample area is selected on
   the map, the scope further narrows to that area.

   Columns: data, particella, area di saggio, specie, D (cm), h (m),
   L10 (mm), pollone (shoot, blank for non-coppice).

   Writers see "+", pencil, and garbage icons (the "+" is enabled
   only when a sample area is selected on the map).

By default section 1 is open; sections 2 and 3 are closed.  Sections
2 and 3 render lazily on first expand.

## Data entry flows

### New rilievo (sample-area visit)

Click "+ Nuovo rilievo" in section 2.  The form modal asks for:

- Parcella (pulldown).
- Sample area: choice between "esistente" (pulldown of sample areas
  in that parcel) and "nuova".  If "nuova": numero (auto-suggested
  next free number for the parcel), lat/lng (shared lat-lng
  component — see below), raggio (default 12 m), gruppo (optional).
- Data (defaults to today).
- Piano di taglio (pulldown, optional — defines which session the
  rilievo belongs to).

On save, an empty `sample` row is created.  The user is dropped into
batch tree entry for that area (see next).

### New tree within an active sample-area selection

With a sample area selected on the map, click "+" in section 3.
Form modal:

- Specie (pulldown).
- D (cm), h (m), L10 (mm).
- Pollone (default 0 for non-coppice; for coppice, auto-increments
  per shoot of the same stump — UI tracks this within the entry
  session).
- Lat/lng of the individual tree (optional — shared lat-lng
  component, defaults to the sample-area center).

The form has the standard two submit buttons: "Salva" returns to the
table; "Salva e aggiungi" leaves the form open for the next tree of
the same sample area.

### Editing / deletion

Pencil opens the row in an edit form; garbage prompts for
confirmation.  Deleting a sample-area row cascades to its tree
records (with explicit warning).  Deleting a tree-row leaves the
sample area intact.

## Lat/lng entry component (shared)

Used here, on Bosco (PAI add), and on Piano di taglio (mark-tree
add).  Two coupled inputs (latitudine, longitudine) plus a "Usa
posizione attuale" button.  The button is enabled only when the
browser's `navigator.geolocation` reports availability and the user
has granted permission; otherwise it is hidden.  On click it
populates the inputs with the device's current coordinates.

Manual entry remains the primary path (office-staff entering data
from paper field notes).

## Cross-page links

- Each particella cell in sections 1 and 3 links to the Bosco
  per-parcel page for that parcel.
- The per-parcel page on Bosco shows recent sampling sessions for
  that parcel and links back here pre-filtered to the relevant
  session.

## URL parameters

TBD — defer until UX is settled.

## Data tables

TBD — defer until UX is settled.  Likely:

- `sessioni.json`: derived rollup of sample sessions.
- `sample_areas.json`: as on Bosco.
- `sampled_trees.json`: denormalized tree+tree_sample+sample for
  table rendering.

## Open questions

1. Name for "sessione".  "Campagne" reads as "countrysides".
   Candidates: *Sessioni*, *Cicli*, *Rilievi-stagione*.
2. Where session-level metadata (notes, official label) lives, if
   anywhere.  Either a small `sampling_session` table keyed on
   (year, plan_id), or no metadata at all (sessions are pure
   derived groupings).
3. Should the region-pulldown selection in section 2 also filter
   sections 1 and 3?  My instinct: yes, the region is a global page
   filter, not just a map zoom helper.
4. Sample-area numbering scheme on creation: parcel-scoped
   monotonic counter (`sample_area.number`), or free-form text?
   Schema says int, parcel-scoped.
5. Multi-shoot coppice trees: the data model uses `shoot=1..N` on
   `tree_sample` keyed to a single `tree_id` representing the stump.
   The UI needs to handle "this is a new shoot of stump X" vs "this
   is a new stump" — likely a "stesso ceppo del precedente" toggle
   in the tree-add form during batch entry.
