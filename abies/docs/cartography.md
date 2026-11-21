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
draw lines and polygons by hand or import points, lines and polygons. These
drawings (disegni) can be edited and promoted to roads or parcels that then
become visible throughout Abies.

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
- Parcels
- Roads
- Drawings

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
note: the modal code should be reused, not duplicated.] A parcel requires a
name and an economic class (**Comparto**). For a new parcel, the dropdown
selects the first configured class in alphabetical order by name (for example,
A). For an existing parcel, it shows the assigned class. Cadastral area may be
undefined (null in the database, rendered as '-' in the UI).

The trash icon on the parcel row asks for confirmation before deleting it.
The icon is muted and disabled on a parcel in use. Saved changes become
visible throughout Abies.

**Confini** makes the parcel's map editable:
- The parcel is highlighted and vertices become visible and movable. Edges
  acquire "insert vertex" controls [Implementation note: as in bosco/ep].
- The **Poligono** tool adds a piece to the highlighted parcel. The new piece
  may be separate from the rest of the parcel and is saved with **Salva**.
- The undo button on the map restores up to ten editing actions.
- The map toolbar offers **Annulla** and **Salva**.

**Annulla** prompts with the standard modal, and then resets the parcel's state
to what it was before the user pressed **Confini**.

**Salva** saves the changes and makes them visible throughout Abies.
Only closed polygons are saved as parcel outlines. If any contours are open,
a confirmation explains that they will be left out. A parcel referenced by
forestry records must keep at least one polygon.

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

**Proprietà** opens a modal to edit the road's name or type. The trash icon on
the road row asks for confirmation before deleting it. Saved changes become
visible throughout Abies.

**Punti** makes the road editable on the map:
- Vertices become visible and movable, and segments provide insert-vertex
  controls [Implementation note: like in bosco/ep].
- The undo button on the map restores up to ten editing actions.
- The map toolbar offers **Annulla** and **Salva**.

**Annulla** prompts with the standard modal, and then resets the road's state
to what it was before the user pressed **Punti**.

**Salva** saves the changes and makes them visible throughout Abies.

Any map changes are persistent across tab navigation.

### Drawing control

The drawing control is a foldable section.

**Nuovo**, next to **Importa…**, opens a drawing session and makes **Linea**
and **Poligono** available on the map, even in an empty compresa. Finishing a
shape adds it to **Disegni** in the selected compresa. The polygon outline stays
open while the operator places vertices. After placing at least three vertices,
clicking the first one closes the outline. The operator can draw a boundary
this way and promote it to the compresa's first parcel. Canceling returns to
the previous selection.

Each drawing has a name and geometry. For imported drawings, the name is
chosen on import. The list includes hand-drawn and imported drawings until
they are deleted or promoted. It has the same hide/unhide affordance as the
other lists, as well as a delete affordance (garbage icon on each row and at
the top). Deletion is gated by the standard warning modal.

The list interacts bidirectionally with the map just like for parcels and roads.

Below the list are the following buttons: **Modifica** **Promuovi**.

**Modifica** makes the drawing editable. The map shows the vertex tools and
allows new lines or polygons to be added. **Annulla** and **Salva** are in the
map toolbar. The map editing tools appear only during an edit.

**Promuovi** opens a modal with the destinations available for the selected
drawing. A line can become a new road. A polygon can become a new parcel or,
if the compresa already has parcels, the operator can choose **Aggiungi a
particella esistente** and select the parcel that will gain the piece. New
roads and parcels use the same property fields as **Proprietà**. The button
is disabled when there is no eligible drawing selected.

The operator can open **Promuovi** while editing. **Annulla** in the modal
returns to the drawing with the edits still on screen. The drawing's own
**Salva** and **Annulla** are still available. Confirming the promotion saves
the new road, new parcel or additional parcel piece and removes the drawing
in one operation. If it fails, the drawing remains available to edit.

**Promuovi** uses the whole drawing as it appears in the editor. In
**Modifica**, the operator can remove unwanted parts and use the rest for one
road or parcel. For example, if a walk follows a road through a gate and then
turns onto a side road, the operator can remove the side-road segment and the
gate point, join the traces on either side of the gate, and inspect the
connection before promoting the road. For a parcel, the operator can keep the
traces that follow its boundary and close the outline.

If the side road is also needed, the operator can copy that segment into a
separate drawing before trimming the walk and promote each drawing in turn.
After promotion, the complete walk is also available for a fresh import under
**Da Ipso → Già importati**.

As always, map changes are persistent across tab navigation.

#### Importing new drawings

Above the drawing list, **Importa…** sits beside **Nuovo**. It opens a modal
where users can add drawings to the current view.

There are three tabs.

- **Da file** opens a file picker for GeoJSON, GPX or georeferenced DXF.
  Waypoints appear as points, each track segment appears as a separate line,
  and the rings of a polygon stay together. ASCII DXF supports points, lines,
  and straight polylines. The EPSG field identifies the source coordinates
  before Abies puts the geometry on the map.
- **Da Ipso** opens on new uploads for this compresa. Previously imported walks
  are available under **Già importati**. Each walk appears once, with its name,
  date and operator. Importing it adds one named drawing to **Disegni**. An
  import from the Ipso inbox can open the same drawing directly. For example,
  if the operator pauses a trace to mark a gate and then resumes, that drawing
  contains the first line, the gate point and the second line, in recording
  order. The gap is visible on the map. **Modifica** lets the operator edit
  the parts individually. Hovering over a recorded point shows its accuracy
  and any note. On a phone, tapping the point opens the same tooltip.
- **Da rilevamento** offers a dropdown of existing tree surveys with mappable
  data in this compresa. Each entry shows the survey name, whether it is free or
  structured, and how many positions are available. Free surveys contribute
  one position for each geolocated tree. Structured surveys use the centers of
  the sample areas visited in that survey.

#### Deleting drawings

The trash icon on a row deletes that drawing. For an Ipso walk this removes
the whole working copy. Removing a single trace segment is an edit within
**Modifica**. The original field recording remains available under **Da Ipso
→ Già importati**, even after the processed upload is cleared from the inbox.
Survey drawings can likewise be imported again from the original survey.
Restoring a drawing imported from a file requires the original file.

The proposed behavior when importing the same walk again is to open its
existing working copy. For example, if the operator corrected the first
segment and removed the second, both decisions are preserved. Once the whole
drawing has been deleted or promoted, importing the walk again creates a
fresh drawing from the complete original recording.

## Working with vertices

The [parcel editor prototype](../../bosco/ep/) is the design starting point. The
operator can place vertices by hand, snap to measured points, or trace a run of
vertices along an existing line or parcel edge. The undo button steps back
through the last ten actions in the current edit, including placing or moving
a vertex, cutting or joining a line, and closing a parcel sketch. A shared
boundary change is one action and undo restores both sides.

A drawing can be reshaped by moving or removing vertices, split at a vertex,
or joined to another line at an endpoint. Three map tools work with vertices.

- **Rimuovi vertice** (trash) removes the selected vertex and reconnects any
  neighbors. A polygon ring must keep at least three vertices. A line can lose
  an endpoint, and removing one end of a single segment leaves a point.
  Removing that point removes the part from the edit.
- **Interrompi al vertice** (broken link) splits a line at an interior vertex,
  which becomes an endpoint of both pieces. On a polygon, it removes the
  selected vertex and leaves its neighbors as the two ends of an open line.
  Other rings remain visible as separate lines.
- **Unisci estremi** (link) displays the endpoints of every line in the edit.
  Selecting two endpoints joins their lines or closes one line into a polygon.
  The tool is available when there are two lines to join or one line with
  at least three distinct vertices to close. In **Punti**, the tool joins
  separate lines into a single line for the road.

Saving a drawing or road after removing all its points asks for confirmation
that the whole feature should be deleted. A road that remains on the map
requires a single line. A parcel in use still requires a closed polygon.

When correcting a shared boundary,  the operator can select the common edge and
show both affected parcels. Moving, inserting or removing a vertex along that
edge changes both outlines together. Vertices at junctions stay in place when
removing them would separate the outlines. Both polygons are validated and
changed in one transaction.

An ordinary boundary correction keeps the parcel identity and its records.
Renaming or splitting a management parcel needs a separate step to review its
history. For a rename, the user is prompted to confirm that the records follow
the same parcel under its new name. For a split, the user is shown the affected
surveys, plans and harvests and asked where they belong before publishing.
Deletion of parcels or roads with dependent records is not allowed.

## Map coverage and background work

Bosco opens on the parcels and masks satellite overlays to the parcel
polygons. The planned imagery and terrain downloads request the bounding
box of all published parcels and roads, plus a margin. An access road can
extend that request while the satellite overlay stays within the parcels.
A viabilità print of the whole compresa includes the road all the way to
its endpoint.

- Changing a road name or type updates the road overlay and its print labels
  or legend. The same elevation and satellite data can be reused.
- The first refresh and published parcel boundary changes check the imagery
  history back to 2015. The worker reuses saved rasters that already cover
  the required area, then rebuilds the parcel mask and statistics.
- Extending a road beyond the parcel bounds grows the required coverage.
  Abies checks the coverage already stored for each imagery date and terrain
  source. If it covers the new extent, no download is needed. If only the
  eastern edge extends beyond it, Abies downloads one strip. An expansion can
  require up to four rectangles. The north and south strips span the new width.
  The west and east strips cover the remaining height, so each corner is fetched
  once. Abies regenerates affected contours across the joins from the assembled
  terrain.

The proposed satellite workflow checks published changes and newly imported
maps hourly. **Aggiorna dati terreni** at the bottom of the sidebar queues an
immediate refresh for the selected compresa and can retry a failed job. The
published vector map remains usable during refresh. The [satellite refresh
proposal](satellite-refresh.md) covers credentials, progress, and seasonal
image discovery.

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
