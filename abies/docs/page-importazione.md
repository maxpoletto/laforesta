# Importazione page

This page reviews Ipso uploads staged by the mobile app and imports valid
sessions into Abies. Ipso uploads never write domain data directly: they create
`IpsoUpload` rows and staged files under `IPSO_INBOX_DIR`, then an authenticated
writer imports or rejects them from this page.

## Upload list

The upper sortable table shows received time, sample/record date, mode,
operator, row count, state, work context, destination, and the error summary.
Imported uploads are hidden by default; the `Anche dati già importati` checkbox
includes them. The search box, sorting, pagination, and export controls follow
the standard table conventions.

Row actions:

- Magnifier: open the upload detail and record preview.
- Pencil: admin-only mode edit before import (`Martellate`, `Rilevamenti
  predefiniti`, `Rilevamenti liberi`, or `Osservazioni`).
- Trash: admin-only staged-upload delete, after the forced `Esporta` download
  step.

The navigation badge counts uploads still in `received` state.

## Detail panel

The lower panel shows session metadata, staged-file errors if files are missing
or corrupt, and a sortable preview of uploaded records. The preview includes
record id, date, parcel, sample area, species, number, diameter/height,
coordinates, and GPS accuracy under `Acc. (m)`. A missing accuracy is shown as
`-`. Observation previews use the same position columns alongside text,
categories, and photo count. Ipso observation position comes from the initial
device GPS fix captured when the observation form opens, with save-time GPS
used only as a fallback; camera photos may also carry device GPS metadata
captured before the camera opens. Ipso ignores photo EXIF GPS.

Target selectors appear only when the current user can import and the upload is
still `received`.

## Import flows

Writers and admins can import received uploads:

- `Martellate`: requires a harvest-plan item destination and creates marked
  trees.
- `Rilevamenti predefiniti`: requires a structured survey destination and
  creates sampled trees via the CSV import core. The Ipso PWA records trees as
  high forest regardless of parcel management.
- `Rilevamenti liberi`: requires an unstructured survey destination and creates
  one null-area sample for the uploaded session; the PWA records these trees as
  high forest too.
- `Osservazioni`: requires no destination and creates observation rows,
  category assignments, and photo metadata/files.

The server validates mode, state, target, staged-file integrity, record ids,
parcel/region consistency, and mode-specific fields. Failed imports leave the
upload staged and store the first error in the upload's error summary.
Successful imports mark the upload `imported`, store importer/timestamp/target
metadata, and leave the staged files in place until an admin deletes the upload.

For predefined and free tree-survey imports, validation errors block the import
and are shown one per line. After validation, the server may return warnings
instead of writing immediately. Warnings are shown in a standard proceed/abort
modal when any row has `h_measured=false`, or when row coordinates fall in a
different parcel from the submitted parcel. Proceeding resubmits the same
import with explicit confirmation; the server recomputes warnings before
writing. The submitted parcel remains authoritative after confirmation.

Target consistency is enforced at import time:

- `Martellate`: every row must match the selected harvest-plan item. For a
  parcel-scoped item, each row must use that parcel. For a region-wide item,
  each row must use a parcel in that region.
- `Rilevamenti predefiniti`: rows must use sample areas in the selected
  survey's grid. If the Ipso session records the survey chosen by the operator,
  importing into a different survey is allowed only when both surveys use the
  same grid.
- `Rilevamenti liberi`: rows carry their own parcel and must target an
  unstructured survey. A single uploaded session creates one
  `Sample(sample_area=NULL)`. Preserved rows store the submitted number as the
  parcel-scoped `preserved_number`.
- `Osservazioni`: rows require text, at least one known category, and
  coordinates. Uploaded photo metadata such as filename and image dimensions
  must match staged photo files.

Rejecting an upload is available to writers and admins only while the upload is
still in the `received` / `Da importare` state.

## Number invariants

Tree numbers are required for predefined sampled trees and free-survey rows
for preserved trees. They are optional for marks and ordinary free-survey rows.
Ipso and the import page preserve submitted values exactly except when an
ordinary free-survey row omits `number`; then import assigns the next
sample-local number.

- `Martellate`: `number` may be null. Ipso proposes the usual next number while
  recording, but the operator may clear it before saving. The staged upload,
  preview, import, and CSV import all preserve a blank number as SQL `NULL`.
- `Rilevamenti predefiniti`: `number` must be a positive integer. Ipso does
  not allow saving/uploading a sampled tree without one, rejects duplicates
  within the selected sample area, and rejects values already present in Abies
  for that survey and sample area. The import page also rejects staged rows
  missing `number`.
- `Rilevamenti liberi`: ordinary rows may omit `number`. Import assigns the
  next number after the explicit ordinary-tree numbers reserved by that
  uploaded session. If a row supplies a number, it must be positive and unused
  within that session/sample; another sample in the same survey may reuse it.
  Preserved rows require a positive submitted number, store it as the
  parcel-scoped `preserved_number`, and receive a separate sample-local
  `number`.

These checks are intentionally repeated at upload/build time and at server import
time: mobile validation gives immediate feedback, while server validation protects
against stale clients, edited staged files, and direct API calls.

## Admin actions

Admins can download a staged upload as a zip file, edit the upload mode before
domain import, or delete the staged upload. Mode edit rewrites `upload.json` and
`upload.sha256`; the original `export.csv` is preserved. Delete removes the
`IpsoUpload` row and staged files only. It does not delete domain records that
were already imported.

All authenticated users can view the inbox and previews.
