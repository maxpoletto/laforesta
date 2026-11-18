# Abies cartography extensions

## Overview

At present, parcel maps and metadata are imported from GeoJSON files at Abies
install/configuration time.

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

Regions can be created or deleted. Creating a new region requires providing a
name and a GeoJSON file (like the current terreni.geojson) describing at least
one parcel.

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
