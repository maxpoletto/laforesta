# acc.py

Forest analysis tool for height/volume calculations and report generation

## Modes

### 1. Generate equations (height-diameter regression curves)
```bash
./acc.py --genera-equazioni --funzione={log,lin} --fonte-altezze={tabelle,ipsometro,originali} \
         --input INPUT_FILE --output EQUATION_FILE --particelle PARCEL_METADATA
```

Fits curves (y = a·ln(x) + b or y = a·x + b) for each (compresa, genere) pair.
Output CSV: `compresa,genere,funzione,a,b,r2,n`

### 2. Calculate heights and volumes
```bash
./acc.py --calcola-altezze-volumi --equazioni EQUATION_FILE --input INPUT_FILE --output OUTPUT_FILE
```

In one pass:
1. Applies height equations to estimate tree heights (unchanged if no equation exists)
2. Computes volume V(m³) for each tree using Tabacchi equations

This ensures heights and volumes are always consistent.

### 3. Generate report
```bash
./acc.py --report --formato={html,latex,pdf} --dati DATA_DIR \
         --particelle PARCEL_METADATA --input TEMPLATE_FILE --output-dir PATH
```

Processes template, substituting `@@directives` with graphs/tables. PDF mode runs pdflatex.
Each directive specifies its data files via `alberi=` and `equazioni=` parameters (relative to `--dati`).

### 4. List parcels
```bash
./acc.py --lista-particelle --particelle PARCEL_METADATA
```

Lists all (compresa, particella) tuples.

## Template Directives

- `@@cd(parameters)` — Diameter class histogram with metadata
- `@@ci(parameters)` — Height-diameter scatter plot with regression curves
- `@@tsv(parameters)` — Volume table with optional confidence intervals

### Parameters

| Parameter | Values | Description | Required | Applicable to |
|-----------|--------|-------------|----------|---------------|
| `alberi=FILE` | filename | Tree data CSV (relative to `--dati`) | **Yes** | all |
| `equazioni=FILE` | filename | Equations CSV (relative to `--dati`) | **Yes** for `@@ci` | `@@ci` only |
| `compresa=NAME` | compresa name | Filter by compresa (default: all) | No | all |
| `particella=NAME` | particella name | Filter by particella (requires compresa) | No | all |
| `genere=GENERE` | species name | Filter by species (default: all) | No | all |
| `per_compresa` | `si`, `no` | Group by compresa (default: `si`) | No | `@@tsv` only |
| `per_particella` | `si`, `no` | Group by particella (default: `si`) | No | `@@tsv` only |
| `per_genere` | `si`, `no` | Group by genere (default: `si`) | No | `@@tsv` only |
| `stime_totali` | `si`, `no` | Show estimated total volumes (default: `no`) | No | `@@tsv` only |
| `intervallo_fiduciario` | `si`, `no` | Show confidence intervals (default: `no`) | No | `@@tsv` only |
| `totali` | `si`, `no` | Add totals row (default: `no`) | No | `@@tsv` only |

**Multi-value parameters**: `alberi`, `equazioni`, `compresa`, `particella`, and `genere` can be repeated:
```
@@cd(alberi=trees1.csv, alberi=trees2.csv, compresa=Serra, compresa=Fabrizia)
@@ci(alberi=trees.csv, equazioni=eq1.csv, equazioni=eq2.csv, compresa=Serra)
```
Multiple `alberi`/`equazioni` files are concatenated; multiple filters are combined with OR.

## File Formats

- **Equation files** (CSV): Regression coefficients for height-diameter relationships
  - Columns: `compresa,genere,funzione,a,b,r2,n`
  - One row per (compresa, genere) pair with sufficient data (n ≥ 10)

- **Height files** (CSV): Field measurements from ipsometer or textbook data
  - Ipsometer: `Compresa,Particella,Area saggio,Genere,D(cm),h(m)`
  - Textbook: `Genere,Classe diametrica,Altezza indicativa`

- **Tree database** (CSV): Complete tree inventory with computed attributes
  - Columns: `Compresa,Particella,Area saggio,Genere,D(cm),h(m),V(m³),Fustaia,Classe diametrica`
  - Heights initially estimated, refined using equations
  - Volumes computed using Tabacchi equations

## Implementation Notes

- **Regression curves** in `@@ci` graphs use `EQUATION_FILE` (not recomputed from current data) to reflect original fit quality
- **Graph x-axis scaling**: Computed per-region from actual filtered data to prevent squashed histograms
- **Volume confidence intervals**: Conservative aggregation (sum of margins) for mixed species
