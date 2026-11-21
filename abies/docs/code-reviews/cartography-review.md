# Cartografia retrospective

Review range `39bfde4a..73cc1a06`. The review follows the editor changes into
the shared Bosco forms and maps, the Ipso upload and reference paths, cadastral
area handling, and the map asset migrations. Reproduction used isolated test
databases and map assets. The findings describe the reviewed revision, and the
status notes record the follow-up. Findings 5–8, audit scope, and the additional
test proposals are pending review.

**P0** marks the satellite identity fix that was prioritized for immediate
resolution. **P1** covers incorrect saved data or misleading map
values in normal use. **P2** covers bounded workflow, interoperability, or
recovery defects. **P3** covers consistency, maintenance, and verification.

## Correctness and recovery

### 1. P1 · Promoting a drawing can create overlapping parcels

Reproduced with two parcels in the same compresa. An existing parcel had a
published polygon. Promoting a copy of that polygon succeeded both as a piece
of the other parcel and as a new parcel. The map ended up with three identical
published outlines. **Confini** rejects a newly overlapping area, so the
outcome depends on which editor action created the polygon.

**Fixed in `6d5d7620`.** Promotion and **Confini** share the overlap check,
including published outlines that have yet to be linked to the parcel catalog.
The message is “Il nuovo confine si sovrappone a un altro poligono”. The
regression tests exercise a new parcel, an addition to an existing parcel,
an unlinked outline, and a valid shared edge.
[Promotion](../../apps/impostazioni/cartography.py#L127)
· [Overlap validation](../../apps/impostazioni/parcel_boundaries.py#L177)
· [Regression tests](../../test/test_cartography.py#L623)

### 2. P0 · Bosco can show satellite values for the wrong parcel

Reproduced by renaming parcel `1` and promoting a different outline to a new
parcel named `1`. The satellite endpoint still returns the old mean for label
`1`. Bosco looks up averages by the current parcel label, so the new parcel
receives the old parcel's values. A boundary correction with the same name
likewise keeps averages calculated over the previous outline.

**Fixed in `0c16e849`.** Bosco looks up averages by stable parcel ID in the
compresa identified by its stable region ID. The averages and mask carry
the same fingerprint of the published parcel IDs and geometries. Abies
checks the fingerprint before serving either product or returning a cache
validation response. Renames preserve the products. Shape and parcel
assignment changes require precompute. Existing installations can rebuild
legacy products with `manage.py build_satellite precompute`.
[Satellite serving](../../apps/bosco/views.py#L935)
· [Regression tests](../../test/test_cartography.py#L421)
· [Satellite data documentation](../page-bosco.md#map-data)

### 3. P1 · Retrying an import can create a duplicate after a lost response

Reproduced in Chromium. The first **Da file** POST committed, then the browser
lost its response. The modal kept the file and displayed the network error.
Pressing **Importa** again submitted a fresh nonce and created a second
drawing. The client creates the nonce inside each call to `write()`, so the
server cannot replay the first result. The same pattern affects other
create-style actions, including survey import and copying a drawing.

**Fixed in `7787da99`.** Retrying the same request after an ambiguous failure
reuses the nonce. A changed destination, action, or request body gets a fresh
token. The Chromium regression drops the response after the server commits,
then checks that the retry uses the same nonce and leaves a single drawing.
[Cartografia transport](../../apps/impostazioni/static/impostazioni/js/cartography.js#L765)
· [Chromium regression](../../test/js/cartography-walk.cjs#L558)
· [Retry contract](../data-architecture.md#retry-receipts-and-write-transactions)

### 4. P2 · Ipso can accept a walk that the office cannot import

Reproduced with both an omitted compresa ID and a compresa ID absent from the
server. The upload endpoint accepted the completed walk. No compresa listed
it under **Da Ipso**, and import into an existing compresa was rejected. For
the absent ID, attempting import to the original destination returned 404.
An operator using cached references after a compresa has been deleted can
reach this state.

**Fixed in `39a58ab1`.** Ipso previously accepted a compresa name even if it
could not resolve the ID. Ipso now requires a positive ID when starting or
uploading a walk, and Abies enforces it at intake. If the recorded ID no
longer resolves, **Importa in Cartografia** in the inbox lets the reviewer
select a current compresa and edit the walk name. This also recovers legacy
walks already in the inbox with a missing ID. Abies preserves the staged
metadata and measurements.
[Session normalization](../../apps/ipso/views.py#L2128)
· [Inbox destination form](../../apps/ipso/static/ipso/js/importazione.js#L436)
· [Regression tests](../../test/test_cartography.py#L161)

### 5. P2 · Reclosing an outline can fill its original holes

Reproduced with the geometry functions. Opening the outside ring of a polygon
with a hole turns both rings into lines. Rejoining the outside endpoints
creates a polygon with only its exterior. The original hole is still
visible as separate linework. **Salva** can then omit the line and publish the
formerly excluded area. In **Confini**, an opened polygon is detached and a
reclosed one is treated as a newly added piece. Reclosing the inner ring on
its own makes a separate polygon that overlaps the exterior.

Preserve ring ownership through cut and join, or offer an explicit way to
restore the holes before saving. Test a polygon with an interior ring through
cut, reclose, undo, and save. Also exercise a reclosed piece sharing an edge
with a neighboring parcel. [Ring breaking](../../apps/impostazioni/static/impostazioni/js/drawing-parts.js#L128)
· [Parcel detach and rejoin](../../apps/impostazioni/static/impostazioni/js/cartography.js#L398)
· [Boundary save validation](../../apps/impostazioni/parcel_boundaries.py#L123)

### 6. P2 · Edited geometry can disagree with the original GeoJSON metadata

Reproduced by moving a vertex beyond the bounds of a bootstrapped feature.
The GeoJSON response contains the new coordinates and the old feature and
collection `bbox` values. Consumers that use those bounds to frame or
index data can miss the changed geometry. The same route normalizes mixed
2D and 3D input to XY for editing, and writes the changed piece from the
flattened geometry. This strips elevations from untouched positions in that
piece as well. That second behavior follows from the save code.

Regenerate or omit saved bounds when geometry changes. Decide whether to
preserve imported altitude through editing or explicitly normalize a map
when taking ownership of it. Add tests for a move outside the old extent and
unchanged coordinates in a mixed-dimension piece.
[Boundary snapshot](../../apps/impostazioni/parcel_boundaries.py#L66)
· [XY normalization](../../apps/impostazioni/drawing_geometry.py#L54)
· [GeoJSON response](../../apps/base/map_data.py#L65)

### 7. P2 · Failed promotion closes the form and loses the entered values

Reproduced in Chromium by promoting a polygon with a duplicate parcel name.
Abies returned a validation error, but the form and the entered locality
were already gone. Drawing naming, copying, and road promotion use the same
dismiss-before-submit pattern. Road and parcel properties and the new import
forms already keep a connected form and show inline errors.

Use one submission lifecycle that keeps values and the map preview while the
operator corrects a rejected form. Test rejection and correction in the same
modal. A version conflict also needs a clear path to review refreshed values.
[Naming and promotion handlers](../../apps/impostazioni/static/impostazioni/js/cartography.js#L508)
· [Inline error handling](../../apps/impostazioni/static/impostazioni/js/cartography.js#L774)
· [Standard form handler](../../apps/base/static/base/js/forms.js#L84)

### 8. P2 · Survey import can use an older position for a revisited tree

Reproduced using **Alberi da preservare**, which is offered as a free survey.
A preserved tree had two dated positions. Bosco's current PAI query selected
the newer one, while the imported drawing contained the older one. The free
survey importer walks rows in primary-key order and keeps the first row per
tree ID. Bosco resolves current PAI records by parcel and preserved number,
using sample date and then row ID. The survey fixture and browser workflow
currently use a single measurement per tree.

Decide which measurement to use when a free survey revisits a tree. For
PAI, either follow Bosco's current-record rule in both the catalog count and
the imported positions or present the historical dataset as such.
[Survey count and position selection](../../apps/impostazioni/cartography_import.py#L226)
· [Current PAI position rule](../../apps/base/preserved_trees.py#L35)

### 9. P2 · The normal predeployment backup cannot undo the map directory migration

**Deferred.** This remains a runbook finding.

Migration `0041` rewrites satellite and print-context metadata and renames
the directories in place. Its reverse operation explicitly requires
restoring the old database and map directories together. The deploy script
runs the standard predeployment backup, which excludes generated map assets.
Restoring that archive and switching to the prior release can therefore leave
existing imagery and print context unavailable to the old reader. This is a
runbook gap identified by inspection. The migration already tests retry and
collision behavior.

Document how to stage and restore the generated directories for this upgrade,
or how to recreate them if rollback is required. Include a restore rehearsal
for a renamed compresa with both kinds of assets.
[Asset migration](../../apps/base/migrations/0041_region_asset_keys.py#L119)
· [Predeployment sequence](../../bin/deploy#L117)
· [Backup scope](../backups.md#L14)

## Localization, UI, and maintainability

- **P3 · Canonical dates and form values.** Dates are displayed as
  `YYYY-MM-DD` by design. Commit `c8c4ee21` uses `unlocalize` for IDs,
  versions, and integer controls in the shared parcel form. The regression
  enables Italian digit grouping and checks the generated values.
  [Parcel form](../../apps/bosco/templates/bosco/_parcel_metadata_form_it.html#L1)
  · [Regression test](../../test/test_cartography.py#L547)
- **P3 · Common labels and form styles.** Commit `7ac746d0` reuses the
  common action and name labels, puts **Annulla** and **Importa** together in
  the file and survey forms, and uses the common affirmative styles for the
  square map buttons. Both Chromium viewports check the import form layout.
  [Cartografia forms](../../apps/impostazioni/templates/impostazioni/_shell_templates_it.html#L329)
  · [UI conventions](../ui-design-patterns.md#bottom-of-form-button-layout)
- **P3 · Disabled-action explanations.** Commit `bb9a1291` adds localized
  tooltips to **Proprietà** and **Confini** for a map-only parcel, and to
  **Promuovi** when the drawing geometry is unsuitable or a road needs an
  active type. Chromium verifies that the promotion tooltip clears after
  the drawing becomes eligible.
  [Button state rules](../../apps/impostazioni/static/impostazioni/js/cartography.js#L230)
- **P3 · Targeted reuse.** Commit `ae4a12e7` shares WGS84 position checks,
  drawing creation, and the selection reset after edits and undo. The
  net change removes eight lines while preserving each import format's
  coordinate rules. The startup follow-up in `4b3e0b8a` loads GeographicLib
  when calculating distances and moves the satellite fingerprint into a
  module shared by the builder and reader. Changes to submission behavior
  are tracked under finding 7. Version snapshots continue to reflect each
  operation's own contract.
  [WGS84 validation](../../apps/base/geodesy.py#L4)
  · [Satellite fingerprint](../../apps/bosco/satellite_data.py)
- **P3 · Make the audit scope explicit.** Published map features are
  versioned and audited. The mutable drawing workspace has an edit version,
  and deletion of the original imported point set removes the recovery copy.
  Both operations leave the audit log unchanged.
  Decide whether working edits are intentionally transient and whether to
  audit the original-source deletion as a small event without duplicating
  the whole GPS payload. [Map models](../../apps/base/models.py#L648)
  · [Original-source deletion](../../apps/impostazioni/cartography.py#L245)
  · [Model convention](../../AGENTS.md#database-model)
- **P3 · Large-compresa measurements.** Commit `40096c6e` benchmarks a
  compresa with 144 parcels and a 25,000-fix walk. Saving one drawing vertex
  sent 778 kB, and the following workspace reload returned 886 kB. Saving
  one shared boundary vertex took a median 50 ms in Django. Ten undo
  snapshots retained 44.6 MiB of extra JavaScript heap.
  [Benchmarks and reproduction command](../cartography-performance.md)

## Test assessment

The existing tests cover desktop and portrait use from Ipso recording through
road publication, geometry editing and undo, region rename and deletion,
map-data migration and revalidation, parcel-use protections, surveyed plot
centers, regional import filtering, and the principal file formats. The
browser workflow is opt-in and combines many features in a single scenario.
Keep that end-to-end check and add smaller cases for the failure and
recovery paths above, so an early UI regression cannot hide a later result.

Good additional targets include delayed responses while switching compresa
or geometry selection, separate road-type filters across Bosco modes, and
map-only legacy outlines alongside published cataloged parcels. Shared
fixtures for the JavaScript boundary preview and Python replay could cover
reversed edges, coincident junctions, holes, and an edit after a detach.
Import fixtures already cover GeoJSON polygon holes and projected DXF
polylines. The malformed-file and geometry cases above provide further
targets.

The full Python suite passed 2,329 tests and skipped the two opt-in Chromium
cases. Both Chromium viewports passed in a separate run, and the JavaScript
suite passed. Six disposable API and domain probes, a Chromium probe for the
lost response and discarded form, and a standalone hole-geometry probe
established reproduction on the reviewed revision. After the fixes through
`40096c6e`, 2,334 Python tests passed, both opt-in Chromium viewports passed
in a separate run, and the JavaScript suite passed. The startup follow-up
passed 113 targeted Python tests and a fresh SQLite migration with
GeographicLib unavailable.
