# UI design patterns

## Objectives

Readable, predictable, discoverable, restful. Low cognitive load,
good use of screen real estate on both desktop and mobile.

## Fonts and colors

Roboto is used throughout.

The UI is strictly two-dimensional: there are no drop-shadows, text inputs are
flat, scroll bars are flat.

Page margins are moderate (40 px) on desktop and small on mobile.

Text inputs have very slightly rounded corners (2-4 px radius).

Buttons have rounded corners (4-8px radius). They are dark green and turn
lighter when hovered over.

Horizontal rules outline the page header as well as collapsible elements (each
page's collapsibles are documented in its `docs/page-*.md`). They are thin
(4px), dark green, and rectangular.

## Export and creation

Many types of objects (survey grids, surveys, harvest plans) can be created and
also exported to CSV. The standard pattern is two side-by-side buttons on the
far right of the work space, [Esporta CSV] (grey) and [+Nuovo X] (green) (where
X is the type of object).

## Tabular data

All tabular data uses sortable-table. All fields are sortable.

Search input sits above the table (left). Debounced (500ms), filters
rows by matching all whitespace-separated terms against the
concatenation of all fields. Sort order is preserved.

Tables fill the viewport height with their own scrollbar. 1px
medium-grey borders; light grey column headers.

Additionally, for users with role "writer" or "admin", tables may (depending on
their semantics — see the relevant page doc (`docs/page-*.md`)) allow
modification:

- Tables that allow row addition have a "+ Aggiungi" button below the bottom
  row, on the right.
- Tables that allow row editing have a "pencil" icon on the right of each row.
- Tables that allow row deletion also have a "garbage can" icon on the right.

## Graphs and charts

All graphs and charts are implemented in Chart.js.

All charts have y-axes that begin at 0.

All color maps range from yellow-green (for low values) to dark green (for high values).

Graphs occupy the full screen width and legends appear below the graph (on both
desktop and mobile).

## Maps

Maps have the following visual structure:

- Optional sidebar on the right (Bosco has one; Campionamenti does not).
  Top of sidebar: status panel + region ("compresa") pulldown (selects and
  zooms to fit). Below: app-specific controls in collapsible sections.
- Upper left (top to bottom): hamburger (sidebar pages only, to show/hide
  sidebar), location pin, zoom +/−, ruler.
- Upper right: basemap selector.

**Basemap selector.**  `BasemapControl` (in `map-common.js`) shows a
64×64 thumbnail of the active basemap (OSM / Topo / Satellite, default
Satellite). Hover/tap expands to show all three; click selects. Added
automatically by `MapCommon.create()` (opt out via
`enableBasemapControl: false`). The `mt=` URL param (`o`/`t`/`s`)
round-trips the choice. On pages with multiple maps, use
`wrapper.syncBasemap(name)` to mirror selection via the
`basemapchange` event without re-firing it.

**Geojson source for parcel polygons.** Parcels are rendered from
`data/geo/terreni.geojson`.  Each polygon feature carries
`properties.layer = "<Compresa>"` (e.g., `"Capistrano"`) and
`properties.name = "<Compresa>-<particella>"` (e.g.,
`"Capistrano-10a"`).  `particelle.geojson` is the QGIS export
companion (outer boundary + viabilità lines) and **must not** be
used as a map source — binding tooltips against it surfaces raw
strings like `"03_FABRIZIA"` from the layer prefix.  After fetching,
call `MapCommon.sortFeaturesByArea(geojson)`
(`apps/base/static/base/js/map-common.js`) so smaller polygons render
and tooltip on top of the larger ones containing them; otherwise the
big "Capistrano" outline can swallow hover events on a nested parcel.

**Parcel hover tooltip.** Use shared helper `parcelLabel(feature)`
(`geo.js`) → `"<Compresa> <particella>"`. Mount with
`bindTooltip(parcelLabel(f), { sticky: true, direction: 'top' })`.

**Parcel style.** Both campionamenti maps share `MapCommon.PARCEL_STYLE`
(warm yellow border, low fill opacity).  Don't inline a style block in
new map renderers — extend or compose the constant so the look stays
consistent across maps and basemaps.

**View persistence across re-renders.** Each section stashes
`savedView: { id, center, zoom }` on every `moveend`/`zoomend`.
On re-render the map restores the view only if the id matches the
current selection (so data writes keep the view, but switching
grid/survey resets to `fitBounds`). Mirror this in any new map.

## Modals

All forms display as 640px-wide overlay modals (`#modal-container`).
They never replace `#content` — the page shell stays intact behind
the darkened overlay, so cancel/dismiss is instant (no rebuild).

Style: slightly rounded corners, thin dark green border, white
background, page darkened ~50%.  Also used for error messages (red
text) and help information.

Infrastructure:

- **`modals.js`**: `show(content)` / `dismiss()` / `showError(msg)` /
  `onDismiss(cb)`.  `onDismiss` registers a one-shot callback that
  fires on dismiss (by Escape or by `dismiss()`); use it to reset
  page state like `inForm = false`.
- **`forms.js`**: `fetchModalForm(url)` fetches a server-rendered
  form fragment and shows it in the overlay modal.
  `renderModalForm(html)` re-renders inside the open modal (for
  validation errors / conflicts).
- **`form-widgets.js`**: shared DOM builders and template-backed
  helpers.  DOM builders: `mkRow`, `mkInput`, `mkTextarea`,
  `mkFileInput`, `mkFormActions`, `mkStatusBox`, `mkErrorsBox`,
  `renderCsvErrors`, `mkCollapsible`, `mkEditDeleteIcons`,
  `mkTabbedModal`.  Template-backed helpers (clone from
  `<template>` elements in `_shell_templates_it.html`):
  `showConfirmModal(message, onConfirm, { confirmLabel })`,
  `showCascadeDeleteModal({ title, warning, exportRequired, onExportCSV,
  onDelete })`.  All pages import from here; never duplicate
  these locally.
- **`csv-export.js`**: `csvField(v, fmt)`, `downloadCSV(lines,
  filename)`, `exportDigest(digest, exportCols, srcCols, filename,
  opts)`.  Shared CSV export primitives; `TableWrapper.exportCSV()`
  is a separate mechanism.

Tabbed modals (e.g., pencil-edit modals with Dettagli + Import tabs)
use `.modal-tabs` / `.modal-tab` / `.modal-tab-body` /
`.modal-tab-bodies` CSS classes (defined in `common.css`).  The
`mkTabbedModal` helper or inline JS measures all tab bodies after
`showModal()` and sets `min-height` on the container to the tallest
tab, so switching tabs never reflows the modal.

### Form card

Every input/edit form renders inside a `<div class="form-card">`
wrapper.  The shared rule in `apps/base/static/base/css/common.css`
sets `max-width: 720px; margin: 0 auto;`, giving the form a centred
column with ample side margins on wide screens.  Use this wrapper
for any new form template.  Do not introduce alternative widths.

The wrapper's `h2` title gets a `12px` bottom margin via
`.form-card h2` so the first row doesn't collapse against the
heading.

### Bottom-of-form button layout

Every form (modal or in-page) ends with a single `.form-actions` flex
row containing the action buttons.  Convention, applied without
exception:

1. **Cancel first (left).** `<button class="btn">{{ S.CANCEL }}</button>`
   — always the leftmost child, label "Annulla".
2. **Primary action (right).** `class="btn btn-primary"`; label
   is the verb that commits state ("Salva", "Crea", "Conferma",
   "Importa"; "Elimina" for destructive flows).
3. **Optional secondary primary (further right).** Typical sets are `[Annulla]
   [Conferma]` and `[Annulla] [Salva] [Salva e continua]`. (Salva e continua
   opens the same modal again after save, for faster batch input.)

Auxiliary buttons that are *not* commits — e.g. the grid planner's
"Pianifica" (recompute the preview lattice) — live separately, near
the controls they relate to, never in the `.form-actions` row.  The
row is reserved for commit / cancel only.

The shared CSS rule is `.form-actions` in
`apps/base/static/base/css/common.css`; nothing else should style the
buttons or their layout.

**Disabled-action affordance.** When an action is impossible in the
current state (e.g., "Elimina" on an area that already has Samples),
disable the button with `[disabled]` and set a `title` attribute that
explains why — never gate the action via a follow-up confirm modal
that simply refuses.  `.btn:disabled` in common.css greys the button
and disables hover-darkening.

### Form row layout

Forms use `.form-row` containing `.form-group` children.
`.form-row` currently uses `display: flex` with `gap: 1rem`; children
get `flex: 1`.  The `narrow` modifier constrains children to `120px`
(`flex: 0 0 120px`).  A future migration to CSS grid is planned.

### Short-entry field width

Short-entry fields — numeric inputs (D, h, L10, quota), lat/lng,
small pull-downs (species, area number), `<input type="date">`,
the "Usa GPS" button — are 120px wide.  Two flavours:

- **Row-level:** `<div class="form-row narrow">` constrains every
  child cell.  Use when ALL cells in the row are short.
- **Per-cell:** `<div class="form-group narrow">` inside a plain
  `<div class="form-row">` constrains just that one cell; siblings
  fill the remaining space.

Wide fields (full-name inputs, description textareas) live in plain
`.form-row` cells without the `narrow` modifier.

### Read-only fields in edit forms

When a form has fields the user cannot change in the current mode
(e.g. N. albero / Specie / Ceduo on the Edit-tree form), render them
as static text using `.readonly-field` (`common.css`) — never as
`disabled` inputs.  Disabled inputs aren't submitted, so they break
the wire format and look interactive-but-broken.  Pair the visible
`<span class="readonly-field">…</span>` with a sibling
`<input type="hidden" name="…" value="…">` so the form still
carries the field on submit.

For read-only header strips (Compresa / Particella / Area di saggio
at the top of tree and mark forms), use `.form-readonly-flat` — an
inline row of `<label>: value` pairs without a card background.
`.form-readonly-block` (grid layout, grey card) is used only by the
area popover on the campionamenti map.

### Deletion-after-export

When attempting to delete certain objects (e.g., a survey), UI presents a user
with a modal warning about the danger of the operation and forces the user to
export data as CSV first. At bottom are three buttons:
[Annulla] [Esporta CSV] [Elimina]
Elimina is disabled until after CSV export.

### Error reporting

Input-related errors are reported directly in the input modals, not in a
separate modal. Errors include validation errors, conflicts, and other
conditions such as network errors.

## `<template>` components

Page scaffolds and client-built modals use HTML `<template>` elements
rendered by Django inside the shell page.  JS clones them via
`cloneTemplate(id)` (`base/js/templates.js`), fills dynamic content,
and wires event handlers.  `<template>` content is inert until cloned.

### File layout

Each app has a `_shell_templates_it.html` included by the shell:

    base/templates/base/_shell_templates_it.html       — shared (confirm, cascade-delete)
    campionamenti/templates/campionamenti/_shell_templates_it.html
    piano_di_taglio/templates/piano_di_taglio/_shell_templates_it.html
    prelievi/templates/prelievi/_shell_templates_it.html

### Conventions

- **`data-section="x"`** on collapsible header/body pairs — JS queries
  by section key to wire toggles and stash refs.
- **`data-action="name"`** on buttons — JS delegates clicks via a
  single `el.addEventListener('click', ...)` handler that dispatches
  by `btn.dataset.action`.
- **`data-field="name"`** on elements whose text is set dynamically
  by JS after cloning (titles, labels, help text).
- **`data-target="name"`** on container elements that JS populates
  with dynamic content (summaries, map hosts, table hosts).
- **`data-role="name"`** on functional elements (forms, sliders)
  that JS needs to query for wiring.
- **`{% if user.role != 'reader' %}`** gates writer-only elements
  (edit/delete icons, add buttons) server-side.  JS `canModify()`
  is no longer needed for elements in templates.
- Static Italian text lives directly in the `_it.html` template
  (the locale is indexed by filename, per the `_it.html` / symlink
  convention).  Dynamic text is set by JS from `S.*` constants.

### What stays in JS

Data-driven content (table rows, chart datasets, map markers),
slider logic, and any structure that depends on runtime state.
Only static scaffolding moves to templates.

## Accessibility

No special accessibility features in v1 (high contrast, etc.).
