# UI design patterns

## Objectives

The objectives of the visual design are:

- Readability: data is presented simply and clearly, with good use of screen
  real estate on both desktop and mobile.
- Predictability: consistent visual guidelines, no unexpected behavior.
- Discoverability: navigation is easy and fast.
- Restfulness: cognitive and visual load are low.

## Fonts and colors

Roboto is used throughout.

The UI is strictly two-dimensional: there are no drop-shadows, text inputs are
flat, scroll bars are flat.

Page margins are moderate (15 px) on desktop and almost disappear (2 px) on mobile.

Text inputs have very slightly rounded corners (2-4 px radius).

Buttons have rounded corners (4-8px radius). They are dark green and turn
lighter when hovered over.

Horizontal rules outline the page header as well as collapsible elements
(each page's collapsibles are documented in its `docs/page-*.md`). They are thin
(4px), dark green, and rectangular.

## Data display overview

Within each domain page, information is displayed consistently in one of three
ways:
- Tabular data is consistently displayed in sortable-table (more on this below).
- Graphical data is displayed using Chart.js.
- Geographic data is displayed using Leaflet.

Some visual elements may be hidden within collapsible sections.

## Tabular data

All tabular data appears in sortable-tables.

- All fields are sortable.
- All tables are searchable via a text input immediately above the table, on the
  left. Search is immediate (no search button) after a 1/2-second sleep to
  debounce rapid keyboard input.
  - Conceptually (not necessarily in actual implementation), the search operates
    as follows:
    1. Split the search text on whitespace.
    1. For every row, join all fields into a single string.
    1. Return all rows that contain all elements of the search text in the given
        order.
  - The search acts purely as a filter. The table does not move, but displays only
    matching rows. Any pre-existing sort order is preserved.
- A table displays rows as far down as the bottom of the viewport. If there are
  more rows, the table has a scrollbar that is separate from the page scrollbar.
  (On mobile, there is enough lateral space to allow the user to also scroll the
  page, not just the table).
- Tables have 1px medium-grey borders and column headers have light grey background.

Additionally, for users with role "writer" or "admin", tables may (depending on their semantics — see the relevant page doc (`docs/page-*.md`)) allow modification:

- Tables that allow row addition have a "+" button below the bottom row, on the
  right.
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

- A navigation bar on the right (sidebar-bearing pages only; Campionamenti
  has no sidebar).
- The following Leaflet tools appear in the upper left corner, top to bottom:
  - A hamburger button to hide/display the navigation bar (sidebar pages only).
  - A location pin.
  - Zoom +/- buttons
  - A ruler
- A basemap selector in the upper right (see below).

The top of the navigation bar (sidebar pages) always contains, top to bottom:
- A status panel
- A pull-down region ("compresa") selector. Choosing a region centers it on the
  map and sets the zoom level to the most detailed level that still includes the
  full region in the viewport.

Below these elements are application-specific controls, organized in collapsible
sections.

This structure derives from the Boscoscopio app (laforesta/bosco/b), with the
basemap selector moved onto the map itself so the pattern works for
sidebar-less maps too (Campionamenti).

**Basemap selector.**  A Leaflet control in the top-right corner of every
map shows the currently active basemap (OSM / Topo / Satellite) as a
64×64 thumbnail.  Hovering it (or tapping the thumbnail on touch
devices) expands the control horizontally to show all three choices;
clicking one selects it and collapses the control back to the active
thumbnail.  Satellite is the default.  The control is added by
`MapCommon.create()` automatically (opt out via
`enableBasemapControl: false`) and works identically on sidebar-bearing
and sidebar-less maps.  The `mt=` URL parameter (`o` / `t` / `s`)
round-trips the choice so basemap selection is bookmarkable; on
Campionamenti, both visible maps share the same setting and update
together via the `basemapchange` Leaflet event that the control fires
on selection.  Use `wrapper.syncBasemap(name)` (returned from
`MapCommon.create`) to mirror a basemap change onto a sibling map
without re-firing the event.  Implementation: `BasemapControl` inside
`apps/base/static/base/js/map-common.js`; CSS rules `.mc-basemap-*` in
`apps/base/static/base/css/common.css`; thumbnail PNGs under
`apps/base/static/base/img/basemap-{osm,topo,sat}.png` (one tile per
provider centred on Capistrano, resized to 128×128).

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

**Parcel hover tooltip.** Every polygon layer binds a sticky tooltip
rendered by the shared helper `parcelLabel(feature)`
(`apps/base/static/base/js/geo.js`), which returns
`"<Compresa> <particella>"`.  Mount with
`bindTooltip(parcelLabel(f), { sticky: true, direction: 'top' })` so
the tooltip follows the cursor inside irregular shapes.  Reuse this
helper in any new map.

**Parcel style.** Both campionamenti maps share `MapCommon.PARCEL_STYLE`
(warm yellow border, low fill opacity).  Don't inline a style block in
new map renderers — extend or compose the constant so the look stays
consistent across maps and basemaps.

**View persistence across re-renders.** Section state objects carry a
`savedView: { gridId | surveyId, center: [lat, lng], zoom } | null`
that the map updates on every Leaflet `moveend` / `zoomend` and
consumes via an `initialView` constructor option.  The stash is
**identity-keyed**: the renderer accepts a `savedView` only when its
id matches the current selection — so `returnToPage`-driven full
re-renders (which transiently clear `activeGridId`) keep the view,
while switching grid/survey resets to `fitBounds`.  This is what
makes "save edit → map stays put" feel right.  Mirror the pattern in
any new map whose contents are re-rendered after data writes.

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
- **`form-widgets.js`**: shared DOM builders for form elements.
  `mkRow`, `mkInput`, `mkTextarea`, `mkFileInput`, `mkFormActions`,
  `mkStatusBox`, `mkErrorsBox`, `renderCsvErrors`, `mkCollapsible`,
  `mkEditDeleteIcons`, `mkTabbedModal`.  All pages import from here;
  never duplicate these locally.

Tabbed modals (e.g., pencil-edit modals with Dettagli + Import tabs)
use `.modal-tabs` / `.modal-tab` / `.modal-tab-body` /
`.modal-tab-bodies` CSS classes (defined in `common.css`).  The
`mkTabbedModal` helper or inline JS measures all tab bodies after
`showModal()` and sets `min-height` on the container to the tallest
tab, so switching tabs never reflows the modal.

## Form card

Every input/edit form renders inside a `<div class="form-card">`
wrapper.  The shared rule in `apps/base/static/base/css/common.css`
sets `max-width: 720px; margin: 0 auto;`, giving the form a centred
column with ample side margins on wide screens.  Use this wrapper
for any new form template.  Do not introduce alternative widths.

The wrapper's `h2` title gets a `12px` bottom margin via
`.form-card h2` so the first row doesn't collapse against the
heading.

## Bottom-of-form button layout

Every form (modal or in-page) ends with a single `.form-actions` flex
row containing the action buttons.  Convention, applied without
exception:

1. **Cancel first (left).** `<button class="btn">{{ S.CANCEL }}</button>`
   — always the leftmost child, label "Annulla".
2. **Primary action last (right).** `class="btn btn-primary"`; label
   is the verb that commits state ("Salva", "Crea", "Conferma",
   "Importa"; "Elimina" for destructive flows).
3. **Optional secondary primary in between.** The typical pair is
   `[Annulla] [Salva] [Salva e continua]` (tree / harvest forms).

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

The secondary "and continue" action is `S.SAVE_AND_CONTINUE`
(`'Salva e continua'`); the legacy `SAVE_AND_ADD` constant has been
removed.

## Short-entry field width

Short-entry fields — numeric inputs (D, h, L10, quota), lat/lng,
small pull-downs (species, area number), `<input type="date">`,
the "Usa GPS" button — are 120px wide.  Two flavours in
`common.css`:

- **Row-level shorthand:** `<div class="form-row narrow">` constrains
  every child cell.  Use when ALL cells in the row are short.
- **Per-cell:** `<div class="form-group narrow">` inside a plain
  `<div class="form-row">` constrains just that one cell; siblings
  keep `flex: 1`.  Use when the row mixes a wide cell (e.g. the
  N. albero pulldown that carries cross-sample-identity options)
  with a short one (Data).  This is what the Edit-vs-New variants
  of the tree form rely on.

Wide fields (full-name inputs, description textareas, the N. albero
pulldown) live in plain `.form-row` cells without the `narrow`
modifier — they flex to fill.  Mixing a wide cell and a narrow
cell in the same row is fine; use the per-cell flavour above.

## Read-only fields in edit forms

When a form has fields the user cannot change in the current mode
(e.g. N. albero / Specie / Ceduo on the Edit-tree form), render them
as static text using `.readonly-field` (`common.css`) — never as
`disabled` inputs.  Disabled inputs aren't submitted, so they break
the wire format and look interactive-but-broken.  Pair the visible
`<span class="readonly-field">…</span>` with a sibling
`<input type="hidden" name="…" value="…">` so the form still
carries the field on submit.

For the Compresa / Particella / Area di saggio header strip at the
top of the Tree form, use `.form-readonly-flat` — a horizontal row
of `<label>: value` pairs without a card background.  Reserve the
legacy `.form-readonly-block` (grey card) for spots where the
read-only context wants to call attention to itself; the flat
variant is the default.

## Accessibility considerations

In its initial version, given the target staff, Abies has no special
accessibility features (high contrast, etc.), though of course enlarging fonts
in the browser is always an option for users.
