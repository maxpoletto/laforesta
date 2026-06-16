# Bootstrap

`bootstrap` loads a canonical data directory into an empty Abies database. The
canonical directory is produced by the legacy converter, checked by the CSV
readers, and then loaded in one transaction.

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

That target resets the local database, migrates, converts the legacy export,
runs `bootstrap`, builds geodata, generates digests, creates the local admin
user, and links templates. `make bootstrap` runs only the bootstrap command
against `$(CANONICAL_DIR)`.

## Contract

Bootstrap has a deliberately narrow contract.

- The database must be empty. The command refuses to run if any guarded model
  already has rows. Reset the database outside this command when a reload is
  needed.
- There is no `--force` mode.
- The whole load runs inside one database transaction.
- Validation errors are accumulated and reported by file; any error prevents
  writes from being committed.
- The input format is canonical. The legacy converter is responsible for fixing
  La Foresta source quirks before bootstrap sees the data.
- Species and product files are optional. If they are missing, built-in defaults
  are seeded.
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
a survey. Species names must match the canonical species table.

`harvest_plan_items.csv` is grouped by `Piano`. A blank parcel means the row is
region-wide. Region-wide rows use `X` as their display parcel in the UI.

`preserved-trees.csv` records PAI/preserved trees and validates parcel,
species, and coordinates for each row.

`harvests.csv` records completed harvest rows. A blank parcel means the row is
region-wide. Dynamic `Specie:<name>` and `Trattore:<name>` columns use canonical
species and tractor names. When `Q.li` is positive, species percentages must sum
to 100. Tractor percentages must be either all blank/zero or sum to 100.

`hypso_params.csv` stores hypsometric parameters by species and eclass.

## Legacy Conversion

Use the converter when starting from the La Foresta export:

```sh
python3 -m ingest.convert_laforesta <legacy_dir> <canonical_dir>
```

The converter is the compatibility layer for legacy formats, names, and derived
canonical columns. Bootstrap should stay strict and simple: it validates and
loads canonical data, but does not try to interpret arbitrary legacy input.
