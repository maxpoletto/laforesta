# Bootstrap

`bootstrap` loads a canonical data directory into an empty Abies database. The
canonical files are checked by the CSV readers and then loaded in one
transaction.

## Commands

```sh
python3 manage.py bootstrap <data_dir>
python3 manage.py bootstrap <data_dir> --check
```

`--check` validates and reports what would be loaded, then rolls back. A normal
run also rolls back if any file has errors.

The development shortcut is:

```sh
make dev
```

That target resets the local database, migrates, runs `bootstrap`, builds
geodata, stages converted mark uploads if present, generates digests,
creates the local admin user, and links templates. `make bootstrap` runs only
the bootstrap command against `$(CANONICAL_DIR)`.

## Contract

Bootstrap has a deliberately narrow contract.

- The database must be empty. The command refuses to run if any guarded model
  already has rows. Reset the database outside this command when a reload is
  needed.
- There is no `--force` mode.
- The whole load runs inside one database transaction.
- Validation errors are accumulated and reported by file. A file is applied
  only if that file has no validation errors; any error anywhere rolls back the
  full transaction.
- The input format is canonical. Legacy-source cleanup happens before files
  reach this command.
- Species and product files are optional. If they are missing, built-in defaults
  are seeded. Species rows may include `Pressler`; blank/missing values use the
  model default of 2.0.
- `terreni.geojson` is part of the full data bundle, but it is not loaded by
  `bootstrap`. Geodata is built afterward by the geodata step.

## Load Order

The command loads files in dependency order:

1. Reference tables: `regions.csv`, `eclasses.csv`, optional `crews.csv`,
   optional `tractors.csv`, optional `species.csv`, optional `products.csv`.
2. Parcels: `particelle.csv`.
3. Containers: `sample_grids.csv`, `harvest_plans.csv`, `surveys.csv`.
4. Sample areas: `sample_areas.csv`.
5. Sampled trees: `sampled-trees.csv`.
6. Hypsometric parameters: `hypso_params.csv`.
7. Harvest plan rows: `harvest_plan_items.csv`.
8. Preserved trees: `preserved-trees.csv`.
9. Harvest rows: `harvests.csv`.

After bootstrap, run the geodata and digest steps, usually through `make geo`
and `make digest` or the broader `make dev` target.

## Canonical Directory

The minimal bootstrap input is:

- `regions.csv`
- `eclasses.csv`
- `particelle.csv`

A complete application data bundle usually also includes:

- `crews.csv`
- `tractors.csv`
- `species.csv`
- `products.csv`
- `sample_grids.csv`
- `sample_areas.csv`
- `surveys.csv`
- `sampled-trees.csv`
- `harvest_plans.csv`
- `harvest_plan_items.csv`
- `preserved-trees.csv`
- `harvests.csv`
- `hypso_params.csv`
- `terreni.geojson`

Optional missing files are skipped, except species and products, which fall back
to built-in defaults.

## File Notes

`particelle.csv` defines parcels and links them to regions and eclasses.
References must already exist in the loaded reference tables.

`sample_grids.csv`, `surveys.csv`, and `harvest_plans.csv` define named
containers. Their child files refer to them by name.

`sample_areas.csv` is grouped by `Griglia`. Each row must reference an existing
sample grid and parcel.

`sampled-trees.csv` is grouped by `Rilevamento`. Tree numbers are unique inside
a survey. Required measurement columns include `D_cm`, `H_m`, `L10_mm`, and
`Pressler`; `Pressler` is stored on `tree_sample.pressler_coeff` and used for
Bosco's percentage volume-increment chart. Species names must match the canonical
species table.

`harvest_plan_items.csv` is grouped by `Piano`. A blank parcel means the row is
region-wide. Region-wide rows use `X` as their display parcel in the UI.

`preserved-trees.csv` records PAI/preserved trees. Required columns are
`Compresa`, `Particella`, `Numero`, `Genere`, `Lon`, and `Lat`; optional columns
include `Data`, `Anno di nascita stimato`, `D_cm`, `H_m`, `H_measured`, `Acc_m`,
`Operatore`, and `Note`. Bootstrap creates both the backing tree and its PAI
observation row, and validates parcel, species, number, and coordinates.

`harvests.csv` records completed harvest rows. A blank parcel means the row is
region-wide. Dynamic `Specie:<name>` and `Trattore:<name>` columns use canonical
species and tractor names. When `Q.li` is positive, species percentages must sum
to 100. Tractor percentages must be either all blank/zero or sum to 100.

`hypso_params.csv` stores hypsometric parameters by species and eclass.

## Canonical Data

Legacy export conversion is handled outside Abies. `bootstrap` intentionally
knows only the canonical files above and defaults to `data/canonical` in local
make targets. Converted mark uploads may be placed in
`data/canonical/marks/*.csv`; `make stage-marks-uploads` stages them
into the Ipso inbox after bootstrap has loaded parcel and species IDs.
