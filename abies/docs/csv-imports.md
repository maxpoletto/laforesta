# CSV imports and diagnostics

CSV files are human-facing input. `apps/base/csv_io.py` owns UTF-8/BOM
handling, header checks, delimiter detection and locale-tolerant numeric
coercion. `apps/base/csv_validation.py:CsvRow` adds field-specific required,
type, range and storage checks. Domain importers retain their reference
lookups, identity rules, duplicate handling and write transactions.

## Error messages

CSV row numbers include the header: the first data row is **Riga 2**. This is
the CSV row number, independent of a tree number or an original Ipso row ID
stored in a recovery column.

Scalar errors name the actual CSV column, or a readable measurement label,
and explain how to correct it. Nonblank rejected values remain in the message
so the user can locate and inspect the original cell. For example, an
uncorrected recovery CSV with no height produces:

> Riga 2: Altezza (h, m): valore obbligatorio mancante.

A malformed diameter is identified as a diameter requiring a positive integer.
A fractional tree number, unrecognized boolean, invalid date, out-of-range
coordinate or unrepresentable coefficient receives its corresponding rule.
Several scalar problems in one row are combined into one message. Reference
and relationship errors identify the unknown parcel/species, conflicting
number/date group, or invalid destination; checks that need a resolved
reference stop when it cannot be resolved.

Optional blank cells retain their domain meaning and defaults. CSV integers
accept integral spreadsheet values such as `12,0`; numeric coercion follows
the detected delimiter/decimal pairing. Storage checks do not round values:
the owning import/domain service still performs its established rounding.
Dates use `AAAA-MM-GG`; mark imports also accept `GG/MM/AAAA`, and the legacy
mark-staging command additionally accepts `GG-MM-AAAA`.

Duplicate headers are rejected before dictionaries can discard a value.
Rows with more cells than declared columns receive a source-row error asking
the user to check separators and quoting. Omitted trailing cells become
blank, so required-field validation can name what is missing. Unknown named
columns remain allowed; recovery metadata columns therefore round-trip
through the ordinary tree importers.

## Rejection and retry

Interactive CSV imports validate before applying a file. Invalid files return
the normal HTTP 400 `validation_error` envelope with localized `errors` and
`message`; no success patches or retry receipt are returned. The existing
form displays the errors, keeps the entered form/file context, and restores
its controls for correction and retry. No valid subset is silently imported.
Equation CSV imports instead show errors in their read-only preview and issue
no confirmation token until the file and its species/region matches are valid.

Ipso's reviewed partial import is a separate workflow. Its required excluded-row
CSV can be corrected and submitted through the ordinary all-or-nothing CSV
importer. A source Ipso session remains completed after its partial import;
retrying the recovery CSV does not reopen that session. See
[Importazione](page-importazione.md#importing-only-valid-rows).

## Sampled-tree identities in CSV

Sampled-tree exports include two optional import columns in addition to
`Albero` (the local tree number):

- **Campione** groups rows into dated samples in a free survey. Values are
  file-local labels, not database IDs; exports number groups from 1. Each
  group must have one date, and tree/shoot numbers must be unique within it.
  Numbers may restart in another group, even on the same date. Without this
  column, or with blank values, all rows form one sample as in a legacy file.
  Structured surveys continue to identify samples by their grid's sample area;
  they do not use this grouping column.
- **Numero PAI** retains the parcel-scoped preserved-tree number independently
  of `Albero`. It must be a positive integer and requires `PAI=true`. When it
  is blank or absent, a PAI row uses `Albero`, retaining the older file format's
  meaning. A preserved number may appear only once per sample and parcel;
  later samples may measure the same preserved tree again.

These columns allow a whole-survey export, or a filtered subset, to be restored
without merging free samples or renumbering preserved trees. Ordinary free-tree
identity remains local to its sample. A PAI identity remains shared across
samples through its parcel and preserved number. Restoring measurements still
requires valid dimensions and dates; exporting a historical unknown height
leaves a blank that the operator must complete before import.

## Import entry points

| Data | Entry point | Validation core |
| --- | --- | --- |
| Structured and free sampled trees | Rilevamenti tree-table Importa CSV; bootstrap | `campionamenti/csv_trees.py` |
| Marked trees | Intervento → Martellate | `piano_di_taglio/views.py:mark_csv_import_view` |
| Harvest records and species/tractor percentages | Prelievi CSV import; bootstrap | `prelievi/csv_harvests.py` |
| Sampling grids and areas | Grid summary-row Importa CSV; bootstrap | `campionamenti/csv_grid.py` |
| High-forest and coppice calendars | Importa CSV in the matching plan section | `piano_di_taglio/csv_plan.py` |
| Canonical harvest-plan items | Bootstrap | `piano_di_taglio/csv_plan.py:validate_canonical_items` |
| Height and volume equations | Impostazioni → Specie → Importa equazioni | `impostazioni/equation_imports.py` |
| Canonical height equations | Bootstrap | `base/hypsometry.py:parse_param_csv` |
| Preserved/PAI trees | Bootstrap | `campionamenti/csv_preserved.py` |
| Parcel metadata | Bootstrap; `update_parcels_from_csv` command | `base/csv_parcels.py` |
| Regions, classes, crews, tractors, species, products and flat named containers | Bootstrap | `base/csv_reference.py` and `base/csv_containers.py` |
| Survey definitions and their default dates | Bootstrap | `base/csv_containers.py:validate_surveys` |
| Legacy mark files converted to staged Ipso sessions | `stage_marks_uploads` command | Shared `CsvRow` fields and mark header aliases |

The command-line paths attach file context to their errors. Bootstrap reuses
the same domain validators rather than maintaining a second parser. The
legacy staging command permits an unknown, unmeasured historical height;
ordinary mark imports require a positive height.

The packaged `apps/base/data/species.csv` is a code-owned default resource,
loaded by `base/refdata.py` through the shared CSV codec. It is separate from
the operator-supplied species/reference CSV handled by `csv_reference.py`.

## Export/import checks

The paired workspace data exports use their domain CSV schemas rather than
formatted table-cell text. Tests export, import into matching destination
contexts, export again and compare values/files for grid areas, sampled and marked trees,
harvest records, both calendar types, and height/volume equations. They cover
both delimiter/decimal pairs (`;` with `,`, and `,` with `.`), quoted text,
nullable fields and identity columns. Sample tests include several free
samples with restarted local numbers and PAI numbers distinct from local ones.

Dendrometric matrix CSVs and other summary reports are report outputs, not
inputs to the tree importers. Historical rows with missing fields that are now
required must be completed before reimport; a backup does not bypass normal
validation. Settings → Specie retains its dedicated equation preview/confirm
workflow. Bootstrap and management-command imports remain command-line entry
points rather than workspace buttons.
