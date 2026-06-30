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

Rejecting an upload is available to writers and admins while the upload has not
already been imported or rejected.

## Admin actions

Admins can download a staged upload as a zip file, edit the upload mode before
domain import, or delete the staged upload. Mode edit rewrites `upload.json` and
`upload.sha256`; the original `export.csv` is preserved. Delete removes the
`IpsoUpload` row and staged files only. It does not delete domain records that
were already imported.

All authenticated users can view the inbox and previews.
