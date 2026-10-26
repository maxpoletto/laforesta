# Osservazioni

## Overview

Osservazioni are geo-referenced field notes with one or more categories and optional photos.
They can enter Abies through Ipso uploads or through the Bosco observations
layer. Final writes always happen on the Abies server; Ipso only stages upload
payloads for later import.

Each observation stores:

- date;
- text description;
- region;
- lat/lon and optional GPS accuracy;
- one or more observation categories;
- zero or more filesystem-backed photos.

## Image policy and upload limits

New permanent photos are resized proportionally to at most **1024 px on the long
axis**, without upscaling, and encoded as JPEG at **quality 85**. EXIF orientation
is applied before resizing. JPEG, PNG, WebP, GIF and HEIC/HEIF inputs must decode
successfully; static JPEG output uses the first frame of animated inputs. HEIF
decoding uses the Pillow HEIF plugin. Original filenames remain in metadata.

Bosco extracts JPEG EXIF GPS and capture time before re-encoding and keeps them
in the photo row. Ipso retains its device-supplied coordinates/time; EXIF GPS
does not override those values. Permanent dimensions, byte size and checksum
describe the stored JPEG. Existing stored photos retain their original bytes.

Input limits are authoritative on the server, including for older/offline Ipso
clients and browser conversion failures:

| Bound | Default | Environment setting |
| --- | --- | --- |
| Raw bytes per photo | 12 MiB | `ABIES_PHOTO_MAX_BYTES` |
| Combined photo bytes per request | 30 MiB | `ABIES_PHOTO_UPLOAD_MAX_BYTES` |
| Decoded pixels per photo | 50 million | `ABIES_PHOTO_MAX_PIXELS` |
| Photos per observation, including retained photos | 20 | `ABIES_OBSERVATION_MAX_PHOTOS` |
| File parts per upload request | 100 | `ABIES_PHOTO_UPLOAD_MAX_COUNT` |

Django's upload handler enforces file bytes/count while receiving multipart
streams, before full buffering. Shared bounded readers also check actual chunks;
claimed sizes/MIME labels/dimensions are insufficient. Pixel bounds precede full
decoding. Malformed/unsupported images are rejected, rather than saved based on
their file signature alone. The Ipso session-body limit remains an additional
bound, published to the PWA in upload configuration.

Limit failures return HTTP 413 with localized guidance; malformed images return
desktop HTTP 400 or native Ipso HTTP 422. The desktop form keeps entered values
and pending file selections on these errors. Rejection creates no observation,
photo row or retry receipt.

## Ipso field capture

PWA version 1.5.4 checks the server-published photo byte/count/pixel limits while
adding photos and before building the final upload payload. Limit failures keep
the session open/editable, preserve local records and display guidance. Older
cached references use the documented default photo limits until refreshed.
The complete estimated multipart/JSON body, including overhead, is also checked
before the session becomes pending-upload.

In Ipso, Osservazioni are a field mode. The operator records observation text,
selects one or more categories from `reference.json`, and attaches photos. The observation
position comes from the initial device GPS fix captured when the observation
form opens; save-time GPS is used only if that initial fix was unavailable.
Photo EXIF GPS is ignored by design: mobile gallery/document pickers do not
reliably expose original EXIF metadata, so Abies treats the device observation
position as authoritative.

Ipso can attach photos from the gallery or by opening the camera. Gallery photos
stay location-free. Camera photos carry the device GPS snapshot captured just
before Ipso opens the camera app, plus a capture-position timestamp. Before
upload, PWA versions since 1.5.3 attempt the same 1024 px / JPEG 85 preparation and report each file
as:

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

Staging retains the received bytes and their checksum so upload retries and
import fingerprints keep their transport identity. Incoming images are decoded
and checked using decoded dimensions before staging. Transmitted dimensions are
retained as part of transport identity (they may already account for EXIF
orientation); permanent dimensions always come from the server's rendition.
Import validates the staged checksum and prepares the permanent JPEG before
opening the domain write transaction. Rendition checksums never replace the
staged upload's identity. This also supports pending sessions created by older
PWA versions, subject to the server input bounds.

The import page validates observation uploads before creating domain rows:

- the upload must still be in the received state;
- the payload checksum and staged photo checksums must match;
- each record must have date, text, lat/lon, and a valid region;
- at least one category id is required and all category ids must exist;
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

Clicking an existing observation opens a standard detail modal with date and
operator on the first row, compresa and formatted position on the second row,
then the correctly singular/plural category label, free text, and photo
thumbnails/links. If multiple visible observations are within 5 m, Bosco shows
one aggregate marker; clicking it opens a chooser modal, and selecting one row
opens the normal detail modal.

Photo thumbnails are captioned by upload-order number; hovering shows filename
and size. If any photo has its own coordinates more than 10 m from the
observation coordinates, the modal also shows a simple ParcelMap section with
parcel borders, the observation marker, and all geolocated photo markers. Photo
markers match the thumbnail captions; photos within 5 m of each other are shown
as a `+` marker whose tooltip lists the grouped photo numbers.

The small images are actual **320 px maximum-long-axis JPEG derivatives**, sized
for the preview cards (up to twice their 160 px display height). Observation
detail metadata exposes a separate `thumbnail_url`; cards and edit-form previews
request that URL, while clicking a card requests the main stored photo. PDF
generation continues to use the main photo.

Thumbnails are generated on demand and cached under
`data/observation-thumbnails/v1-320/`, keyed by source path/checksum. This covers
photos already stored before the resizing policy without rewriting their bytes
or database metadata. Stored legacy files are streamed from disk and retain the
decoded-pixel guard; incoming-upload byte caps do not block these previews.
Both thumbnail and main-photo endpoints require login and use the existing
protected-file/conditional-response behavior. Cache hits avoid re-decoding;
missing or undecodable originals return 404. Thumbnail files are regenerable
cache data and do not need to be restored from backup. Concurrent generation
uses unique temporary files and atomic publication.

The detail title row also contains `Genera PDF`. The browser downloads a report
named `osservazione_<yyyymmdd>_<region>_<category>.pdf`. The report contains a
`Rapporto <categoria>` title, date, compresa, formatted observation position,
free text, photos in two columns, and an overview OSM map with only the
observation point marked. If a photo cannot be loaded, report generation skips
that optional image; if OSM tiles cannot be drawn, the map falls back to a
simple parcel-geometry overview.

Writers also see `Modifica` and `Elimina`. `Modifica` opens the same form with
title `Modifica osservazione`; in edit mode the region is a selectable pulldown.
`Elimina` uses the shared confirmation modal and optimistic-lock delete flow.

The add/edit form contains, in order:

1. region and date;
2. required description text;
3. category checkboxes, with at least one category required;
4. photo section;
5. shared lat/lon entry fields plus the `Usa GPS` button when browser
   geolocation is available.

When photos are added through the Bosco web form, JPEG EXIF GPS coordinates are
read on the server and stored as per-photo metadata when present. Missing or
malformed EXIF data is ignored and does not block the upload.

Existing photos are shown in upload order. The photo section has a `+ Aggiungi`
button that opens the browser file picker. Each existing or newly selected photo
has a small `x` button in the upper-right corner. Removing a photo only changes
the pending form state; the database row and stored file are deleted only after a
successful save.

Save and delete responses use the same digest patch/delete envelope as other
Bosco write flows. On success, the client updates the cached `observations`
digest and rerenders the current observations layer without a full page reload.
The multipart add/edit submission carries an idempotency nonce. Repeating the
same successful submission replays its original response rather than storing
another observation/photo or conflicting with the already-applied edit.

## Photo backup and recovery

Runtime backups include main observation photos and the staged Ipso inbox, which
can contain uploaded photo bytes. The thumbnail directory is a regenerable cache.
Media-changing requests participate in a cross-process snapshot lock through
their database commit/rollback and filesystem hooks. Backup briefly pauses those
mutations while copying the database and media to a private snapshot, then
releases the lock before verification and compression. Photo GETs do not acquire
the lock. Deleting a live photo after capture cannot remove it from that copy.

Before publishing a backup, the script checks every photo referenced by the
captured database against the copied file, byte size and SHA-256 checksum.
Missing/corrupt photos fail the backup instead of producing a successful archive
with dangling references. Restore repeats these checks in a staging directory
before exposing its destination. Restore main photos and the database from the
same archive; thumbnails regenerate on demand. See [Backups](backups.md) for the
archive format, full restore procedure and remaining operational checks.
