# pdg-2026: Piano di Gestione

Generates the 2026 forest management plan document from field survey data. Produces PDF (via LaTeX), HTML, and CSV outputs by expanding template directives (`@@grafico_classi_diametriche`, `@@tabella_classi_diametriche`, `@@ci`, etc.) with computed forest statistics.

# Key Files

- `pdg/` — Main package (refactored from former `acc.py` monolith):
  - `core.py` — Entry point: report template expansion, top-level orchestration.
  - `computation.py` — Data models (`ParcelData`, `ParcelStats`), regression fitting, volume/area computation.
  - `simulation.py` — Harvest simulation engine, growth tables, `harvest_parcel`, `schedule_harvests`.
  - `io.py` — CSV/data loading and writing.
  - `formatters.py` — Output formatting (numbers, tables, snippets).
  - `harvest_rules.py` — Harvest policy logic: volume-based and age-based rules per comparto (A–F). Ceduo parcels (comparto F) always return (0, 0).
- `inc.py` — Growth projection model: Pressler's formula, transition matrices.
- `template/` — Report templates (HTML, LaTeX, CSV) containing `@@` directives that `pdg/core.py` expands.

# Build Pipeline

All data flows through `bosco/data/`: raw CSVs in, calculated CSVs and reports out.

```bash
make all            # equazioni → altvol → pdf
make equazioni      # fit height-diameter curves (3 sources: originali, ipsometro, tabelle)
make altvol         # calculate tree heights and volumes (Pressler coeff 200)
make incrementi     # compute growth increments (separate from main pipeline)
make html           # generate HTML report
make pdf            # LaTeX → PDF (pdflatex 3-pass + biber)
make csv            # generate harvest plan CSV
make test           # pytest --cov=acc
```

# Domain Constants

- `SAMPLE_AREA_HA = 0.125` — fixed sample plot size
- `MATURE_THRESHOLD = 20` — diameter (cm) below which trees are excluded from harvest calculations
- `MIN_TREES_PER_HA = 0.5` — minimum density for diameter class graphs
- Diameter classes: 5-cm buckets
- Pressler coefficient: 200
