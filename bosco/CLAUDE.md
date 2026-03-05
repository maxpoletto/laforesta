# Bosco: Forest Management Application

Interactive web application for forest management operations. Currently client-only (vanilla JS + Leaflet); will evolve into a full-stack app with a Python backend.

# Web Apps

Each subdirectory is a self-contained single-page app:

- `b/` — Boscoscopio: parcel property viewer with satellite overlay
- `q/` — Database browser
- `ep/` — Parcel editor (manual geometry editing)
- `pac/` — Sampling planner
- `ads/` — Sample areas viewer
- `pai/` — Indefinite-growth trees viewer
- `cdt/` — Harvest calendar

Shared assets (map utilities, UI components, styles) live in `a/`.

# Data

All apps load data from `data/` via fetch:

- `alberi.csv`, `particelle.csv`, `alsometrie.csv` — tree and parcel data
- `particelle.geojson`, `terreni.geojson` — parcel geometries
- `satellite/` — Sentinel-2 imagery per region, organized as `{region}/{date}/`
- `produzione/` — Historical production timeseries (JSON)

# Utility Scripts (`util/`)

Data processing pipeline for satellite and production data:

```bash
cd util && make dates       # find cloud-free Sentinel-2 dates
cd util && make fetch       # download imagery from Copernicus
cd util && make precompute  # compute vegetation indices (NDVI, NDMI, EVI)
cd util && make production  # aggregate historical production data
cd util && make sync        # rsync satellite + production data to laforesta.it
cd util && make test        # run sentinel and production tests
```

Requires env vars `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET` for Copernicus authentication.

Regions are parameterized: currently Serra, Fabrizia, Capistrano. Cloud thresholds: summer (Jun–Jul) 1%, winter (Jan–Feb) 10%.
