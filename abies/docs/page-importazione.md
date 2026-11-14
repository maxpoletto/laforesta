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
coordinates, and GPS accuracy under `Acc. (m)`. A missing height or accuracy
is shown as `-`. Observation previews use the same position columns alongside
text,
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

The server validates mode, state, target, staged-file integrity, record IDs,
parcel/region consistency, and mode-specific fields before writing. All tree
imports require a positive height. A calculated height is valid even though
`h_measured=false`; a missing height is an error and is never replaced by zero.
An uploaded session can still be reviewed after an administrator changes its
mode, so conversion from marks to samples exposes any fields that the selected
sample destination requires.

### Reviewing tree errors

For Martellate, predefined samples and free samples, the server checks the
complete session and identifies the valid rows and the excluded rows. The
review modal separates two labeled sections: **Errori:** lists the excluded
rows in a red panel, one source row per line; **Avvertenze:** lists nonblocking
warnings about the importable rows. A section is hidden when it has no messages.
Long error and warning lists scroll within their own panels.

Beneath the messages, three lines report **Righe totali: X**, **Righe escluse a
causa di errore: Y**, and **Righe importabili: X-Y**. These count source rows,
not individual messages: a row with several errors is excluded only once.
The safety instruction and action buttons follow the counts.

A message such as **Riga 30: altezza mancante.** uses the original Ipso row ID
shown in the preview. It does not refer to array position or the row's current
position after sorting the preview table.

Errors explain the specific invalid fields: date, diameter, height, GPS values,
sample numbers, references, or calculated values that cannot be stored. Tree
numbers must also fit the shared integer storage bounds; out-of-range values
are excluded before any domain insert. Several
measurement problems on a row are listed together. Conflicting tree-number or
sample-date groups are excluded as a group rather than accepting an arbitrary
first row.

L10 is optional. Null or blank means no core was measured; a supplied
measurement must be a positive integer width in millimeters. Zero is also
recognized as the internal missing-value marker used by the mobile payload
and sample storage. Negative, fractional, malformed and out-of-range values
are errors, reported with an instruction to leave L10 blank or enter a
positive integer. Ipso has no L10 entry control.

Errors concerning the entire operation, such as corrupt/missing staged files,
an invalid destination or a closed worksite, prevent the whole import. There
is no valid subset to accept in those cases. Observation imports also retain
their whole-session transaction, including their photos.

### Importing only valid rows

The review provides **Annulla**, **Esporta righe escluse**, and **Importa righe
valide (N)**. The instruction above them reads:

> Per sicurezza, esporta le righe escluse prima di importare quelle valide.

The import button starts disabled. A successful CSV export enables it; a
failed export leaves it disabled and lets the user retry. With no valid rows,
the button remains disabled after export too. Canceling writes no forestry
data and leaves the upload available for review or rejection.

For survey imports, unmeasured-but-present heights and GPS/parcel disagreements
among valid rows remain nonblocking warnings. They appear in the same review,
and importing the valid subset confirms those warnings too. The supplied
parcel remains authoritative. If a session has warnings but no row errors,
it uses the ordinary proceed/abort warning modal.

The server checks the destination again before writing. For example, another
user could have imported a conflicting tree number while the review was open.
If the destination or accepted/excluded selection changed, a fresh review is
shown and the export gate resets. The rejected-row CSV and final confirmation
must describe the same reviewed set. If the destination was deleted, the
operation stops with an error and imports no rows.

### Completion and CSV recovery

Importing the accepted subset completes the Ipso session. Its state is
`imported`, displayed as **Importato parzialmente** when rows were excluded.
The detail panel shows imported/excluded counts and retains **Esporta righe
escluse** for writers and admins. The session cannot be imported again or have
its mode changed. Retrying the same completed request returns the original
result and does not add duplicate trees. Completed partial sessions follow
the existing **Anche dati già importati** visibility filter.

The stored import result records which source rows were accepted and which
were excluded, including their error reasons. Accepted domain rows, the audited
result, terminal session state and retry receipt commit together. If the write
fails, none of that accepted subset remains imported. The original staged JSON
and files remain until an administrator explicitly deletes the upload.

The recovery CSV contains only excluded rows. Its columns use the normal
Martellate or sample CSV format and add **Riga Ipso**, **Motivi esclusione**, and
**Dati originali Ipso**. The last column preserves the complete original JSON
record, including metadata not represented by ordinary domain CSV columns.
Blanks remain blank and decimal numbers use the install locale.

To recover excluded measurements, correct the CSV and import it through the
normal Martellate or survey CSV importer into the intended destination. Those
importers validate the corrected data as usual. Ordinary free-survey rows may
keep a blank tree number; their CSV importer assigns a sample-local number.
Preserved rows still require their original positive PAI number. There is no
second Ipso import of the partially completed session.

### Destination rules

Target consistency is enforced at import time:

- `Martellate`: every row must match the selected harvest-plan item. For a
  parcel-scoped item, each row must use that parcel. For a region-wide item,
  each row must use a parcel in that region. Every imported row needs a
  positive measured or calculated height. The preview displays missing heights
  as `-`; rejected-row CSVs preserve them as blank fields for correction.
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
