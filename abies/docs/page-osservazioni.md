# Osservazioni

## Overview

Osservazioni are geo-referenced field notes with optional categories and photos.
They can enter Abies through Ipso uploads or through the Bosco observations
layer. Final writes always happen on the Abies server; Ipso only stages upload
payloads for later import.

Each observation stores:

- date;
- text description;
- region;
- lat/lon and optional GPS accuracy;
- zero or more observation categories;
- zero or more filesystem-backed photos.

## Ipso field capture

In Ipso, Osservazioni are a field mode. The operator records observation text,
selects categories from `reference.json`, and attaches photos. The observation
position comes from the device GPS fix held by the PWA. Photo EXIF GPS is ignored
by design: mobile gallery/document pickers do not reliably expose original EXIF
metadata, so Abies treats the device observation position as authoritative.

Ipso can attach photos from the gallery or by opening the camera. Before upload,
the PWA attempts to resize/recompress large images and reports each file as:

```
filename · original-size -> upload-size · status
```

where `status` is `ok` or an error. Ipso refuses uploads whose estimated multipart
body exceeds the configured upload limit.

## Upload and import

Ipso uploads are staged under `IPSO_INBOX_DIR` as an `IpsoUpload` row plus
filesystem payload files. Observation uploads use multipart form data: the JSON
payload lists records and photo metadata, while each photo is sent as a separate
part keyed by its client photo id.

The import page validates observation uploads before creating domain rows:

- the upload must still be in the received state;
- the payload checksum and staged photo checksums must match;
- each record must have date, text, lat/lon, and a valid region;
- category ids must exist;
- photo content must be a supported raster type;
- import fingerprints must not duplicate an already imported observation.

Successful import creates `Observation`, `ObservationCategoryAssignment`, and
`ObservationPhoto` rows, stores photos under `OBSERVATION_MEDIA_DIR`, marks the
observations digest stale, and records the upload as imported. Failed imports
leave staged data unchanged so the operator can retry or reject the upload.

## Bosco observations layer

Bosco has an `Osservazioni` map mode. It shows dark-green point markers for
observations assigned to the selected region. Parcel geometry is used only to add
parcel context when a coordinate falls inside a known parcel. Category filters
list all active categories with in-scope counts; year filters include every year
from the first to the last in-scope observation.

Writers see a `+ Aggiungi` button. Clicking it opens the standard modal form
`Nuova osservazione` with the current region fixed and lat/lon blank. Clicking a
blank point on the map in observations mode opens the same modal with lat/lon
pre-filled from the clicked coordinate.

Clicking an existing observation opens a standard detail modal with date, region,
text, categories, coordinates, GPS accuracy, operator, and photo thumbnails/links.
Writers also see `Modifica` and `Elimina`. `Modifica` opens the same form with
title `Modifica osservazione`; in edit mode the region is a selectable pulldown.
`Elimina` uses the shared confirmation modal and optimistic-lock delete flow.

The add/edit form contains, in order:

1. region and date;
2. required description text;
3. category checkboxes;
4. photo section;
5. shared lat/lon entry fields plus the `Usa GPS` button when browser
   geolocation is available.

Existing photos are shown in upload order. The photo section has a `+ Aggiungi`
button that opens the browser file picker. Each existing or newly selected photo
has a small `x` button in the upper-right corner. Removing a photo only changes
the pending form state; the database row and stored file are deleted only after a
successful save.

Save and delete responses use the same digest patch/delete envelope as other
Bosco write flows. On success, the client updates the cached `observations`
digest and rerenders the current observations layer without a full page reload.
