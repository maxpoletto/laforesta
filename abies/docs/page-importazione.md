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
- Pencil: admin-only mode edit before import (`Martellate`, `Campionamenti`,
  or `PAI`).
- Trash: admin-only staged-upload delete, after the forced `Esporta` download
  step.

The navigation badge counts uploads still in `received` state.

## Detail panel

The lower panel shows session metadata, staged-file errors if files are missing
or corrupt, and a sortable preview of uploaded records. The preview includes
record id, date, parcel, sample area, species, number, diameter/height,
coordinates, and GPS accuracy when present.

Target selectors appear only when the current user can import and the upload is
still `received`.

## Import flows

Writers and admins can import received uploads:

- `Martellate`: requires a harvest-plan item destination and creates marked
  trees.
- `Campionamenti`: requires a survey destination and creates sampled trees via
  the CSV import core.
- `PAI`: imports preserved trees without a destination selector.

The server validates mode, state, target, staged-file integrity, record ids,
parcel/region consistency, and mode-specific fields. Failed imports leave the
upload staged and store the first error in the upload's error summary.
Successful imports mark the upload `imported`, store importer/timestamp/target
metadata, and leave the staged files in place until an admin deletes the upload.

Target consistency is enforced at import time:

- `Martellate`: every row must match the selected harvest-plan item. For a
  parcel-scoped item, each row must use that parcel. For a region-wide item,
  each row must use a parcel in that region.
- `Campionamenti`: rows must use sample areas in the selected survey's grid. If
  the Ipso session records the survey chosen by the operator, importing into a
  different survey is allowed only when both surveys use the same grid.
- `PAI`: rows carry their own parcel and no target selector is shown. Each
  imported row becomes a preserved-tree sample row whose `number` is the
  submitted sample-local sequence and whose `preserved_number` is the
  parcel-scoped PAI number.

Rejecting an upload is available to writers and admins only while the upload is
still in the `received` / `Da importare` state.

## Number invariants

Tree numbers are required for preserved trees (PAI) and sampled trees, but
optional for marks. Ipso and the import page preserve submitted values exactly;
they do not fill in missing numbers during import.

- `Martellate`: `number` may be null. Ipso proposes the usual next number while
  recording, but the operator may clear it before saving. The staged upload,
  preview, import, and CSV import all preserve a blank number as SQL `NULL`.
- `Campionamenti`: `number` must be a positive integer. Ipso does not allow
  saving/uploading a sampled tree without one, rejects duplicates within the
  selected sample area, and rejects values already present in Abies for that
  survey and sample area. The import page also rejects staged rows missing
  `number`.
- `PAI`: `number` must be a positive integer. Ipso proposes the next value for
  the selected parcel, does not allow clearing it, rejects duplicates within the
  upload for that parcel, and rejects values already present in Abies for that
  parcel. On import this value is stored as `tree_sample.preserved_number`; the
  row also has a sample-local `tree_sample.number`, initially the same value.
  The import page rejects staged rows missing, invalid, non-positive, or
  duplicate within-parcel `number` values.

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
