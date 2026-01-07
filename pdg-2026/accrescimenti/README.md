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

### 2. Calculate heights
```bash
./acc.py --calcola-altezze --equazioni EQUATION_FILE --alberi INPUT_FILE --output OUTPUT_FILE
```

Applies equations to estimate tree heights. Heights unchanged if no equation exists.

### 3. Calculate volumes
```bash
./acc.py --calcola-volumi --alberi INPUT_FILE --output OUTPUT_FILE
```

Computes volume V(m³) for each tree using Tabacchi equations. Stores in output CSV.

### 4. Generate report
```bash
./acc.py --report --formato={html,latex,pdf} --equazioni EQUATION_FILE \
         --alberi TREE_DATABASE --particelle PARCEL_METADATA \
         --input TEMPLATE_FILE --output-dir PATH
```

Processes template, substituting `@@directives` with graphs/tables. PDF mode runs pdflatex.

### 5. List parcels
```bash
./acc.py --lista-particelle --particelle PARCEL_METADATA
```

Lists all (compresa, particella) tuples.

## Template Directives

- `@@cd(parameters)` — Diameter class histogram with metadata
- `@@ci(parameters)` — Height-diameter scatter plot with regression curves
- `@@tsv(parameters)` — Volume table with optional confidence intervals

### Parameters (all optional)

| Parameter | Values | Description | Applicable to |
|-----------|--------|-------------|---------------|
| `compresa=NAME` | compresa name | Filter by compresa (default: all) | all |
| `particella=NAME` | particella name | Filter by particella (requires compresa) | all |
| `genere=GENERE` | species name | Filter by species (default: all species on one graph) | all |
| `per_particella` | `si`, `no` | Group by particella (default: `si`) | `@@tsv` only |
| `stime_totali` | `si`, `no` | Show estimated total volumes (default: `no`) | `@@tsv` only |
| `intervallo_fiduciario` | `si`, `no` | Show confidence intervals (default: `no`) | `@@tsv` only |
| `totali` | `si`, `no` | Add totals row (default: `no`) | `@@tsv` only |

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
