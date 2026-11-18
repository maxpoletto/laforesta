# Install Abies

## Fresh local instance

From the directory containing `manage.py`, use Python 3.14 and a local filesystem
that supports SQLite WAL.

1. Install dependencies and create the local configuration.

   ```sh
   python3.14 -m venv .venv
   . .venv/bin/activate
   python -m pip install --require-hashes -r requirements.txt
   cp compose/.env.local.example compose/.env.local
   ```

2. Set the site and brand names in `compose/.env.local`. Place `logo.png`,
   `favicon.gif` and `ipso-logo.gif` in `data/branding/`. Set
   `ABIES_IPSO_SECRET` if operators will use the Ipso app.

3. Create the database, link language-specific templates (currently only
   Italian), and create the first admin. The admin command prompts for credentials.

   ```sh
   make sync-branding migrate link-templates admin
   ```

4. Choose the initial data.

   - **Empty map.** Continue to step 5. Use **Impostazioni → Cartografia**
     to draw or import regions.
   - **Map file only.** Run `make geo MAP_GEOJSON=/path/to/map.geojson`.
     Supply a GeoJSON FeatureCollection of WGS84 Polygons or MultiPolygons
     with coordinates in longitude, latitude order. Put the region name in
     `properties.layer` and the parcel label in `properties.name`, for example
     `"Capistrano"` and `"Capistrano-01b"`. Several features may belong to
     the same parcel. Bosco lists a region once you import its parcel records
     from the CSV files below. Abies matches each boundary to its record by
     region and parcel name.
   - **Existing forestry records.** Place `regions.csv` (comprese) and
     `particelle.csv` (parcels) in `data/canonical/`. Parcels can use the
     six default economic classes (A–F). To supply your own classes, include
     `eclasses.csv`. Include `map.geojson` there if you have boundaries.
     Run `make load-canonical-data` to import them together. File formats and
     optional records are described in the [bootstrap guide](docs/bootstrap.md).

5. Run `make serve`, open `http://127.0.0.1:8000/` and sign in. To inspect
   the installed map, open `/api/geo/map.geojson` while signed in.

## Hosted instance

Follow the [host provisioning and deployment instructions](README.md#remote-dev-and-production)
for TLS, Apache, Docker, branding, per-instance secrets and backup mounts.
Once the instance is provisioned, import forestry records with
`make bootstrap-dev` or `make bootstrap-prod`. To import just the map, run
`python manage.py build_geo /app/data/canonical/map.geojson` inside the
instance container. Configure [scheduled backups](docs/backups.md).
