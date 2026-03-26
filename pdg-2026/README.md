# pdg.py

Forest analysis tool for height/volume calculations and report generation.

## Modes

### 1. Generate equations (height-diameter regression curves)
```bash
./pdg.py --genera-equazioni --funzione={log,lin} --fonte-altezze={tabelle,ipsometro,originali} \
         --input INPUT_FILE --output EQUATION_FILE --particelle PARCEL_METADATA
```

Fits curves (y = a·ln(x) + b or y = a·x + b) for each (compresa, genere) pair.
Output CSV: `compresa,genere,funzione,a,b,r2,n`

### 2. Calculate heights and volumes
```bash
./pdg.py --calcola-altezze-volumi --equazioni EQUATION_FILE --input INPUT_FILE --output OUTPUT_FILE \
         [--coeff-pressler VALUE]
```

In one pass:
1. Applies height equations to estimate tree heights (unchanged if no equation exists)
2. Computes volume V(m³) for each tree using Tabacchi equations

`--coeff-pressler` overrides the Pressler coefficient for all trees.

### 3. Calculate growth increments
```bash
./pdg.py --calcola-incrementi --input INPUT_FILE --output OUTPUT_FILE
```

Computes percentage growth increment (IP) for each tree and writes the result to output CSV.

### 4. Generate report
```bash
./pdg.py --report --formato={csv,html,tex,pdf} --dati DATA_DIR \
         --particelle PARCEL_METADATA --input TEMPLATE_FILE --output-dir PATH \
         [--ometti-generi-sconosciuti] [--non-rigenerare-grafici] \
         [--separatore-decimale {punto,virgola}]
```

Processes template, substituting `@@directives` with graphs/tables. PDF mode runs pdflatex.
Each directive specifies its data files via `alberi=` and `equazioni=` parameters (relative to `--dati`).

Options:
- `--ometti-generi-sconosciuti`: omit species with no equations from graphs
- `--non-rigenerare-grafici`: skip regenerating existing graph files
- `--separatore-decimale`: decimal separator in output (`virgola` = Italian style, default)

### 5. List parcels
```bash
./pdg.py --lista-particelle --particelle PARCEL_METADATA
```

Lists all (compresa, particella) tuples.

## Template Directives

### Graphs
- `@@grafico_classi_diametriche(params)` — Diameter class histogram
- `@@grafico_classi_ipsometriche(params)` — Height-diameter scatter plot with regression curves
- `@@grafico_incremento_percentuale(params)` — Percentage growth graph by diameter class

### Tables
- `@@volumi(params)` — Volume table with optional confidence intervals
- `@@tabella_classi_diametriche(params)` — Diameter class table
- `@@tabella_incremento_percentuale(params)` — Percentage growth table
- `@@prelievi(params)` — Harvest table based on comparto/age rules
- `@@piano_di_taglio(params)` — Multi-period harvest plan (cutting schedule simulation)

### Structural
- `@@particelle(compresa=X, modello=BASENAME)` — Expand a sub-template for each parcel in a compresa. The sub-template can use `@@compresa` and `@@particella` as placeholders, and may contain further directives.
- `@@prop(compresa=X, particella=Y)` — Insert parcel properties (metadata) inline

### Common Parameters

| Parameter | Values | Description | Required |
|-----------|--------|-------------|----------|
| `alberi=FILE` | filename | Tree data CSV (relative to `--dati`) | **Yes** (all except `@@prop`, `@@particelle`) |
| `equazioni=FILE` | filename | Equations CSV (relative to `--dati`) | **Yes** for `@@grafico_classi_ipsometriche` |
| `compresa=NAME` | compresa name | Filter by compresa (default: all) | No |
| `particella=NAME` | particella name | Filter by particella (requires compresa) | No |
| `genere=GENERE` | species name | Filter by species (default: all) | No |
| `per_compresa` | `si`, `no` | Group by compresa (default: `si`) | No |
| `per_particella` | `si`, `no` | Group by particella (default: `si`) | No |
| `per_genere` | `si`, `no` | Group by genere (default varies by directive) | No |
| `totali` | `si`, `no` | Add totals row (default: `no`) | No |
| `stime_totali` | `si`, `no` | Use estimated totals scaled to parcel area (default: `si`) | No |

**Multi-value parameters**: `alberi`, `equazioni`, `compresa`, `particella`, and `genere` can be repeated:
```
@@grafico_classi_diametriche(alberi=trees1.csv, alberi=trees2.csv, compresa=Serra, compresa=Fabrizia)
@@grafico_classi_ipsometriche(alberi=trees.csv, equazioni=eq1.csv, equazioni=eq2.csv, compresa=Serra)
```
Multiple `alberi`/`equazioni` files are concatenated; multiple filters are combined with OR.

### Graph Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `metrica` | see below | Metric to plot (default varies by directive) |
| `stile` | freeform | CSS class (HTML) or `\includegraphics` options (LaTeX) |
| `x_max` | integer | Override x-axis maximum (0 = auto) |
| `y_max` | integer | Override y-axis maximum (0 = auto) |

`@@grafico_classi_diametriche` metrics: `alberi_ha`, `G_ha`, `volume_ha`, `alberi_tot`, `G_tot`, `volume_tot`, `altezza` (default: `alberi_ha`).

`@@grafico_incremento_percentuale` metrics: `ip`, `ic` (default: `ip`). `per_compresa` and `per_particella` default to `no`.

### `@@volumi` Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `intervallo_fiduciario` | `si`, `no` | Show confidence intervals (default: `no`) |
| `solo_mature` | `si`, `no` | Only include mature trees (D > 20 cm) (default: `no`) |

### `@@prelievi` Parameters

Harvest rules are defined in `harvest_rules.py` (comparto-based limits from volume and age tables).

| Parameter | Values | Description |
|-----------|--------|-------------|
| `col_comparto` | `si`, `no` | Show comparto column (default: `si`) |
| `col_eta` | `si`, `no` | Show mean age column (default: `si`) |
| `col_area_ha` | `si`, `no` | Show area in hectares (default: `si`) |
| `col_volume` | `si`, `no` | Show total volume (default: `no`) |
| `col_volume_ha` | `si`, `no` | Show volume per hectare (default: `no`) |
| `col_volume_mature` | `si`, `no` | Show mature-tree volume (default: `si`) |
| `col_volume_mature_ha` | `si`, `no` | Show mature-tree volume per hectare (default: `si`) |
| `col_pp_max` | `si`, `no` | Show PP_max % (default: `si`) |
| `col_prelievo_ha` | `si`, `no` | Show harvest per hectare (default: `si`) |
| `col_prelievo` | `si`, `no` | Show total harvest (default: `si`) |

Note: the `genere` filter is not allowed — use `per_genere=si` to group by species.

### `@@piano_di_taglio` Parameters

| Parameter | Values | Description | Required |
|-----------|--------|-------------|----------|
| `volume_obiettivo` | number | Target standing volume (m³/ha) for the plan | **Yes** |
| `anno_inizio` | year | First harvest year (default: 2026) | No |
| `anno_fine` | year | Last harvest year (default: 2040) | No |
| `intervallo` | years | Harvest interval (default: 10) | No |
| `mortalita` | fraction | Annual mortality rate (default: 0) | No |
| `calendario=FILE` | filename | Past harvests CSV (relative to `--dati`) | No |
| `col_comparto` | `si`, `no` | Show comparto column (default: `si`) | No |
| `col_eta` | `si`, `no` | Show mean age column (default: `si`) | No |
| `col_pp_max` | `si`, `no` | Show PP_max % (default: `si`) | No |
| `col_prima_dopo` | `si`, `no` | Show before/after volumes (default: `si`, requires `per_particella=si`) | No |

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

- **Regression curves** in `@@grafico_classi_ipsometriche` use the equations file (not recomputed from current data) to reflect original fit quality
- **Volume confidence intervals**: Conservative aggregation (sum of margins) for mixed species
- **Harvest rules** are encoded in `harvest_rules.py` as a function of comparto, age, volume, and basal area
