# Abies cartography extensions

## Overview

Parcel boundaries now live in SQLite and are imported once from an optional
GeoJSON file. The current La Foresta map has 99 polygon features for 85 parcels.
Each feature has its own version and history, so the editor can update both
sides of a shared boundary in one transaction and leave other features alone.
At this scale the median compact geometry is 1,412 bytes. Per-vertex identity
would additionally require ring membership, ordering and rules for shared
edges. Revisit database topology when shared-edge validation or indexed
spatial queries become a significant server workload.

Installation and the map file format are described in [INSTALL.md](../INSTALL.md).

We want to extend Abies to (a) support dynamically adding features, including
regions (i.e., it should be possible to do an Abies "cold start" with no
preconfigured regions), parcels, and roads and paths; (b) allow editing of
existing features; (c) allow import of data from GeoJSON, GPX, and georeferenced
DXF (*); (d) allow manual tracing over data from those formats, as well as over
geolocated samples inside Abies.

Additionally, we want to support a 1-m resolution LiDAR DTM base layer.

(*) It may be acceptable to provide utilities to convert GPX and DXF to GeoJSON.

## Viewing

Current Bosco features will continue to exist, but we want to be able to display
the DTM base layer as well. An initial implementation might display DTM
elevations using colors, as a heat map. A more advanced implementation would
provide a rotatable/zoomable 3D map, e.g., using Deck.gl tile-3D layers. It
would be especially compelling to overlay features like roads and boundaries
onto this 3D view. If we implement the 3D view, it might make sense to migrate
Abies entirely to Deck.gl, while keeping Leaflet for Ipso.

Additionally, given the added presence of roads, we want all Bosco views to be
able to hide or display roads. There are likely to be several different kinds of
roads: for example, primary roads, secondary roads, logging trails, and
footpaths. The set of road types should be configurable, similarly to the set of
observation types today, and it should be possible to hide or display the
different kinds independently.

## Abstractions

A tentative proposal for useful absractions:

* Points: an object with a lat/lon, e.g., an observation, or a sampled tree.
* Polylines: naturally map to roads or GPX tracks.
* Polygons: closed polylines, naturally map to parcels.
* Layers: especially useful for imports: for instance, we want to be able to
  import one or more GPX tracks onto one layer, and then trace a road by hand in
  another layer using those tracks as guidance.

## Editing

Editing cartographic information will not happen in Bosco, but on its own page,
possibly reachable from the settings page.

### Operations on regions

Regions can be created or deleted. The planned editor should support a named region drawn by hand or
initialized from GeoJSON, including a cold start without cadastral data.

The region editor should queue background generation of the satellite layers
and printable map context when a region first has boundaries or those
boundaries change. A fresh install that imports a map should use the same
backend job. The region should be usable while the layers are being built.

A region may be deleted only if it is not used in any harvest plan (including
marks and harvests) or survey.

### Operations on parcels

All operations on parcels happen within the scope of a region.

Parcels may be created, removed, or have their borders changed.

A parcel may only be removed if it does not appear in any harvest plan
(including marks or harvests) or survey.

A parcel may be created by importing GeoJSON or DXF data, or it may be drawn by
hand. If it directly borders other parcels, it may share vertices along the
bordering edge (no gaps).

Parcel borders may be edited by hand. (A parcel is a polygon; see below.)

### Operations on roads and road types

All operations on roads happen within the scope of a region.

Road types may be created, removed, or renamed. Roads types have an editable
sort order attribute (similar to observation types) that can be used to govern
their display (by convention, more important roads have lower sort order). A
road type may only be removed if it is not in use.

Roads may be created, removed, or edited. A road may be created by importing
GeoJSON, DXF, or GPX data, or it may be drawn by hand.

Roads may be edited by hand. (A road is a polyline; see below.)

### General operations on polylines and polygons

See laforesta/bosco/ep/ for an early prototype of a parcel editor that was used to
generate the base data for the La Foresta s.r.l. production instance of Abies.
Many of the ideas in this section are based on that prototype.

Roads are represented as polylines, parcels as polygons. Imported GPX tracks are
converted to polylines. GeoJSON and DXF objects are converted to polylines or
polygons depending on their geometry.

Operations on polylines and polygons:
* move any vertex of a polyline or polygon;
* split a polyline into two polylines by deleting a vertex;
* merge two polylines into one by connecting two endpoints;
* break a polygon into a polyline by removing a vertex;
* convert a polyline into a polygon by joining two endpoints;
* create a new polygon or polyline based on contiguous sequences of vertices of
  one or more polygons or polylines.

A new "raw" polygon or polyline needs to be explicitly converted to a road (with
a name and type) or parcel (with a name and management type (coppice or high forest)).

### Role of layers

Users can create new temporary layers, particularly  for tracing. It probably
makes sense to allow layers to be reordered. Only the topmost (highest z-index)
layer is editable.

It is not obvious whether the explicit concept of layers should be exposed when
viewing data (i.e., in Bosco). On the one hand, it is natural to say "hide the
road layer", and it would make it natural to map, e.g., each different
characteristic onto a different layer. On the other hand, it is an additional
concept or abstraction that is not domain-specific and that operators would need
to learn.

## Cartography editor sketch

The page has a layout like the Bosco, with map on the left and a sidebar on the
right.

Map:
- Map controls are as in Bosco, with the addition of line and polygon tool as in
  laforesta/bosco/ep.
- We may consider a bounded undo/redo feature in the future.

Sidebar:
- A region selector similar to Bosco's, but with an affordance to add a new
  region, rename a region, or delete a region (as long as it has no data).
- Lists of parcels and roads for the currently selected region.
- An affordance (file selector and import button?) to import data (GeoJSON, GPX,
  DXF). When data is imported, it appears as simple lines or polygons on the
  map. If the coordinates are off the current view port, the map zooms out
  appropriately.
- An affordance to promote the currently selected line or polygon to road or
  parcel, respectively, and to delete the current line or polygon.
- An affordance to clear this parcel's map of all lines or polygons that are not
  roads or parcels.
- An affordance to edit the currently selected road (name, type) or parcel
  (name, economic class, cadastral area), and to delete roads and parcels (as
  long as they have no associated data).
- An affordance to "publish" the current region. This fetches satellite data and
  map context (and, in the future, DTM) for every parcel in the region and makes
  it available in Bosco. A region can of course be published multiple times,
  after any change.
