# Cartography in Abies

A new Abies instance should also be able to start with an empty map and record
its first parcel and region (compresa) through the UI.

Additionally, operators must be able to edit parcel boundaries and maintain maps
of forest roads and trails.

Parcel outlines already live in SQLite. Bosco, Ipso and map builders read them
as GeoJSON. Road types and economic classes are configurable in **Impostazioni →
Dati di base**. The editor described below is proposed for **Impostazioni →
Cartografia**. See the [installation guide](../INSTALL.md) for what can be
imported today.

## General layout and concepts

The cartography editor has a layout similar to the Bosco page, with a map on the
left and a sidebar on the right. On a phone, the sidebar can open as a panel and
keep the current selection when the operator returns to the map.

Users can edit existing parcels (particelle) and roads (strade). They can also
import points, lines, and polygons from various sources. These we call drawings
(disegni). They may be edited, and upgraded to parcels and roads that are then
"published" and made visible to the rest of the app.

Parcels and roads belong to regions (comprese). The only required feature of
region is its name. A region's parcels may be non-contiguous. A region does not
have a boundary distinct from the boundaries of its parcels. Opening a region
fits all its parcels, roads, and/or drawings on the map.

An operator can publish an empty compresa. Its ID and name are included in Ipso
reference data so it appears in the selector after the next sync, so the
operator can then walk parcel boundaries or roads with the phone running Ipso.
Imagery and terrain update jobs can wait until a published road or parcel
establishes an extent.

## Sidebar overview

The sidebar has 4 main sections:

- Region selector
- Drawings
- Parcels
- Roads

## Region selector

The region selector is at the top of the sidebar. It lists current regions in
alphabetical order in a pulldown.

The selected region is sticky across tab navigation (i.e., if the operator
momentarily switches to another part of Abies).

Below the selector are two buttons: [modifica] [nuova]

Both open modals with standard Abies style (e.g., like Piano di Taglio > Nuovo),
with fields for the name and an optional textual description and "Annulla" and
"Salva" buttons.

If the region has only map content, **Modifica** also offers a red **Elimina**
button. The confirmation names the compresa and counts its parcels, outlines,
roads and drawings. Confirming removes those items with the compresa. A region
used in a harvest plan, survey or other forestry record is protected. This
includes observations, saved Ipso walks and field uploads awaiting import.

## Drawing and selecting

The sidebar lists **Particelle**, **Strade** and **Disegni** for the selected
compresa. Clicking an object on the map selects its entry in the sidebar and
vice versa. Drawings can be points, lines or polygons. Drawings are grouped by
source so that an operator can hide a dense survey or show it again while
tracing. New drawings always appear under **Disegni**. There is no layer order
to manage.

Three conversion buttons remain visible in place. Each is clearly disabled
whenever the selection cannot be used for that action.

| Selected drawing | Actions enabled |
| --- | --- |
| One line | **Crea strada** |
| One polygon | **Crea particella**, **Aggiungi a particella esistente** |
| A point, several objects, an existing road or parcel, or nothing | None |

**Crea strada** asks for the road name and type. **Crea particella** asks for
the parcel name and economic class, which determines whether it is coppice.
Cadastral area should be optional. The map also shows an area calculated from
geometry, clearly labeled as such. **Aggiungi a particella esistente** asks
which parcel owns the additional piece and is disabled until the compresa has a
parcel to choose. These actions create draft edits ready for publication.

The operator can delete an individual drawing. **Elimina disegni** clears all
remaining drawings in the selected compresa. Its confirmation names the
compresa and gives the count. Both draft and published roads and parcels are
kept, as are the original field measurements.

### Working with vertices

The [parcel editor prototype](../../bosco/ep/) is the design starting point. The
operator can place vertices by hand, snap to measured points, or trace a run of
vertices along an existing line or parcel edge. While drawing, offer undo last
point, finish and cancel. Full undo and redo can wait.

A drawing can be reshaped by moving vertices, split at a vertex, or joined to
another line at an endpoint. Closing a line makes a polygon. Opening a polygon
makes a line. Keep disconnected pieces and interior holes when editing parcels.

When correcting a shared boundary,  the operator can select the common edge and
show both affected parcels. Moving or inserting a vertex changes both outlines
together. Both polygons are validated and changed in one transaction, so a strip
of land cannot accidentally disappear between them. Where surveyed edges are
close but do not coincide, the operator can choose the edge to follow.

An ordinary boundary correction keeps the parcel identity and its records.
Renaming or splitting a management parcel needs a separate step to review its
history. For a rename, the user is prompted to confirm that the records follow
the same parcel under its new name. For a split, the user is shown the affected
surveys, plans and harvests and asked where they belong before publishing.
Deletion of parcels or roads with dependent records is not allowed.

## Importing drawings

The **Importa** menu offers three ways to add drawings to the current compresa.
After an import, the editor reports how many objects it added and brings them
into view if they were offscreen. The operator can inspect or clear the drawings
before using them. A separate import preview is a possible later addition.

- **Da file** opens a file picker for GeoJSON, GPX or georeferenced DXF.
  Waypoints appear as points, each track segment appears as a separate line,
  and the rings of a polygon stay together. If the coordinate system of a DXF
  file is unknown, the editor asks for it before placing the drawing.
- **Da Ipso** opens on new uploads for this compresa. Previously imported point
  sets are available under **Già importati**. Each session appears once, with
  its name, date and operator. An import from the Ipso inbox can open the same
  drawings directly. Drawings appear in the order they were recorded. A trace
  interrupted by a manual point appears as a line, a point and a second line.
  Hovering over a point shows its reported accuracy and any note. On a phone,
  tapping the point opens the same tooltip.
- **Da rilevamento** offers a dropdown of existing tree surveys with mappable
  data in this compresa. Each entry shows the survey name, whether it is free
  or structured, and how many positions are available. The editor identifies
  measurements by their recorded parcel or sample area. A position just outside
  an incorrect boundary is therefore still available to help correct it.

For a free survey, the editor imports individual tree positions and reports how
many measurements lack coordinates. A structured survey contributes one point
at the center of each visited plot, labeled with the sample area number. In
either case the operator chooses which points to connect and in what order
when tracing. Both survey types are described on the
[survey page](page-campionamenti.md).

On the first import of an Ipso point set, Abies saves the original fixes,
segments, notes and accuracy under the session ID. The operator can then delete
the processed upload from the inbox, along with its staging files. The saved
set remains available in **Da Ipso**. An upload discarded before import leaves
no saved set.

**Elimina disegni** clears the working copies from the map and sidebar.
Choosing the set again in **Da Ipso** restores missing drawings and preserves
any edits to drawings still present. When the operator no longer needs the
saved source, a separate delete action in the picker removes just that source.
Survey imports can likewise identify previously imported points by their survey
and measurement or plot IDs. The [Ipso guide](ipso-abies.md) describes how
point sets are recorded and uploaded.

## Publishing a map

**Pubblica** applies the pending road and parcel edits for one compresa. It is
an explicit action even for a single moved road vertex. The operator can leave
the editor and return to the draft. Before publication, the editor lists the
roads and parcels that will be added, changed or removed. Raw drawings can
stay alongside them for the next visit.

While an edit or deletion is pending, other maps display the last published
version. For example, if an operator moves and renames a road, Bosco continues
to show it in its original position under its original name until **Pubblica**
succeeds.

Each `MapFeature` has a `published` flag. The migration marks existing features
as published, including outlines imported before their parcel catalog. The
shared GeoJSON reader serves only published features to Bosco, Ipso and the
other maps.

A draft holds both geometry and catalog changes. At publication, Abies checks
that the versions the operator edited are still current and applies the changes
in one transaction. SQLite stores geometry per feature, so a correction to a
shared edge updates only the affected pieces.

### Map coverage and background work

Bosco opens on the parcels. Imagery and terrain also cover the roads leading
to them. The required coverage for a compresa is the bounding box of all its
published parcels and roads, plus a margin. An access road can therefore
extend coverage without changing Bosco’s opening view. A viabilità print of
the whole compresa includes the road all the way to its endpoint.

- Changing a road name or type updates the road overlay and its print labels
  or legend. The same elevation and satellite data can be reused.
- Correcting a parcel edge updates the affected parcel masks and satellite
  statistics, even if the imagery already covers the corrected parcel.
- Extending a road beyond the parcel bounds grows the required coverage.
  Abies checks the coverage already stored for each imagery date and terrain
  source. If it covers the new extent, no download is needed. If only the
  eastern edge extends beyond it, Abies downloads one strip. An expansion can
  require up to four rectangles. The north and south strips span the new width.
  The west and east strips cover the remaining height, so each corner is fetched
  once. Abies regenerates affected contours across the joins from the assembled
  terrain.

These jobs run in the background after publication and after a new installation
imports its initial map. The published vector map is usable while they run.
The operator can see their progress and retry a failed job without publishing
again.

## Bosco and printed maps

In Bosco, operators can show or hide each configured road type independently.
Forestry staff maintain these roads inside Abies, including logging trails
absent from OSM. Road markings that are part of a raster basemap remain
visible when an Abies road type is hidden. A plain terrain basemap appears in
the upper-right selector when one is available. The viabilità print can show
every Abies road and trail together with labeled contours derived from a
1 m digital terrain model (DTM) (future work).

Hillshade from that terrain model would also help staff read the terrain on
screen. Elevation coloring and a rotatable 3D view with roads and parcels
overlaid are later possibilities. The first print map can use the existing
[contour rendering and label placement](ui-maps.md#local-vector-context).

## First end-to-end test

In the first test, an operator walks a road while recording a trace in Ipso.
In the editor, the operator imports the point set, adjusts the line, chooses
**Crea strada**, assigns the road type and publishes the road. The road then
appears in Bosco and can be hidden with the filter for its type. Back in the
editor, the operator clears the drawings and recovers the original walk from
**Da Ipso**. Survey import and shared parcel boundary corrections are also
part of the first version.
