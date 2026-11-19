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
(disegni). They may be edited and upgraded to parcels and roads that then become
visible throughout Abies.

Parcels and roads belong to regions (comprese). The only required feature of
region is its name. A region's parcels may be non-contiguous. A region does not
have a boundary distinct from the boundaries of its parcels. Opening a region
fits all its parcels, roads, and/or drawings on the map.

An operator can create and save an empty compresa. Its ID and name are included
in Ipso reference data so it appears in the selector after the next sync, so the
operator can then walk parcel boundaries or roads with the phone running Ipso.
Imagery and terrain update jobs can wait until a published road or parcel
establishes an extent.

## Sidebar overview

The sidebar has 4 main sections:

- Region selector
- Drawings
- Parcels
- Roads

### Region selector

The region selector is at the top of the sidebar. It lists current regions in
alphabetical order in a pulldown.

The selected region is sticky across tab navigation (i.e., if the operator
momentarily switches to another part of Abies).

Below the selector are two buttons: **Proprietà** (grey) **Nuova** (green)

Both open modals with standard Abies style (e.g., like Piano di Taglio > Nuovo),
with fields for the name and an optional textual description and "Annulla" and
"Salva" buttons.

If the region has only map content, **Proprietà** also offers a red **Elimina**
button. The confirmation names the compresa and counts its parcels, outlines,
roads and drawings. Confirming removes those items with the compresa. A region
used in a harvest plan, survey or other forestry record is protected. This
includes observations, saved Ipso walks and field uploads awaiting import.

### Parcel control

The parcel control is a non-foldable section.

The top part is a list of the region's parcels sorted in natural order (1, 2a,
2b, ..., 9, 10, ...), or "Nessuna particella" if there are none. The list is
variable height and up to approximately 300 px tall, enough for 6 full rows.
[Implementation note: lists here and in the Bosco view should all have the same
stretch behavior, height, and font size, an Abies constant.]

Every row of the parcel list contains a parcel name and an "eye" icon for
hiding/unhiding that parcel from the map. Above the list, aligned with the eye
icon, is another eye icon that hides/unhides all parcels.

The parcel list interacts with the map: clicking a parcel on the list selects
the parcel on the map, and vice-versa.

Below the list are two buttons that become enabled when a parcel is selected:
**Proprietà** **Confini**

**Proprietà** opens a modal identical to that used in the Bosco parcel detail
page, with the addition that the parcel name is editable also. [Implementation
note: the modal code should be reused, not duplicated.] The only required
property is name. In particular, cadastral area may be undefined initially (null
in the database, rendered as '-' in the UI).

If the parcel is still unused, **Proprietà** also offers a red **Elimina**
button, analogously to the region workflow. On save or delete, changes
immediately becoming visible throughout Abies.

**Confini** makes the parcel's map editable:
- The parcel is highlighted and vertices become visible and movable. Edges
  acquire "insert vertex" controls [Implementation note: as in bosco/ep].
- A "new polygon" tool appears on the map, and can be used to add a new region
  to the parcel.
- A "undo" button appears on the map, to undo just the last step.
- Two buttons appear below **Proprietà** and **Confini**: **Annulla** (grey) **Salva** (green)

**Annulla** prompts with the standard modal, and then resets the parcel's state
to what it was before the user pressed **Confini**.

**Salva** saves the changes and makes them visible throughout Abies.

Any map changes are persistent across tab navigation.

If a user edits a shared edge, both parcels are highlighted in the UI and
changes to both parcels are saved together (atomically) or canceled together. 

### Road control

The top part is a list of the region's roads sorted in natural order, or
"Nessuna strada" if there are none, laid out identically to the parcel list.

The road list has the same hide/unhide affordance as the parcel list.

The road list interacts with the map: clicking a road on the list selects
the road on the map, and vice-versa.

Below the list are two buttons that become enabled when a road is selected:
**Proprietà** **Punti**

**Proprietà** opens a standard modal where the user can edit the road's name or
its type, or delete it. Changes immediately become visible throughout Abies.

**Punti** makes the road editable on the map:
- Vertices become visible and movable, and segments provide insert-vertex
  controls [Implementation note: like in bosco/ep].
- A "undo" button appears on the map, to undo just the last step.
- Two buttons appear below **Proprietà** and **Punti**: **Annulla** **Salva**

**Annulla** prompts with the standard modal, and then resets the road's state
to what it was before the user pressed **Punti**.

**Salva** saves the changes and makes them visible throughout Abies.

Any map changes are persistent across tab navigation.

### Drawing control

The drawing control is a foldable section.

Drawing's attributes are their name (chosen on import) and their geometry.

The top part is a list of drawings that have been imported but neither deleted
nor promoted to roads or parcels. The list has the same hide/unhide affordance
as the other lists, as well as a delete affordance (garbage icon on each row and
at the top). Deletion is gated by the standard warning modal.

The list interacts bidirectionally with the map just like for parcels and roads.

Below the list are the following buttons: **Modifica** **Promuovi**.

**Modifica** makes the drawing editable, and makes visible  the usual editing
map controls and **Annulla** and **Salva** buttons.

**Promuovi** open a modal that allows the user to save the drawing as a road
(for polylines) or a parcel (for polygons). The modals are exactly those used to
edit the **Proprietà** of the two types of objects.

Clicking **Promuovi** before **Salva** during an ongoing edit is allowed and
implicitly saves the result. So then clicking **Annulla** in the modal still
preserves the drawing changes.

After a promotion, the drawing is removed from the set of drawings, and the
corresponding road or parcel is added to appropriate list.

As always, map changes are persistent across tab navigation.

#### Importing new drawings

Within the foldable drawing control, below the drawing list and its controls, is
an **Importa ...** button. It opens a modal where users can add drawings to the
current view.

There are three optins:

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
  data in this compresa. Each entry shows the survey name, whether it is free or
  structured, and how many positions are available. Free surveys contribute
  positions for each geolocated tree, structured surveys for each sample area.

#### Deleting drawings

As mentioned, drawings can be deleted from the drawing list.

Deleting a GPX or DXF remove it from Abies entirely.

Deleting an Ipso import or a survey deletes the drawing from the current
region's drawing list, but does not delete the collected data. The survey
remains unchanged, and the Ipso import remains visible in the Import tab. Both
types of objects could be reimported as drawings.

## Working with vertices

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

## Map coverage and background work

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
