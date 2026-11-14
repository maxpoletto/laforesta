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

## Import entry points

| Data | Entry point | Validation core |
| --- | --- | --- |
| Structured and free sampled trees | Campionamenti survey editor; bootstrap | `campionamenti/csv_trees.py` |
| Marked trees | Intervento → Martellate | `piano_di_taglio/views.py:mark_csv_import_view` |
| Harvest records and species/tractor percentages | Prelievi CSV import; bootstrap | `prelievi/csv_harvests.py` |
| Sampling grids and areas | Campionamenti grid editor; bootstrap | `campionamenti/csv_grid.py` |
| High-forest and coppice calendars | Harvest-plan editor | `piano_di_taglio/csv_plan.py` |
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
