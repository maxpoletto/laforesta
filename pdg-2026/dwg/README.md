# DXF/DWG Parcel Extraction Project

## Overview
Extracting geographic parcel boundaries from Italian cadastral DWG files
for display via online mapping tools. The workflow converts DWG → DXF → GeoJSON.

## Coordinate Systems
- **Common source:** WGS84 UTM Zone 33N (EPSG:32633)
- **Target:** WGS84 lat/lon (EPSG:4326)
- **Other potential sources:** Italian cadastral data often uses ED50/UTM 33N (EPSG:23033) or Monte Mario (EPSG:3004), causing 50-150m datum shifts. Different layers may use different datums.

Common Italian CRS options:
| Name | EPSG | Notes |
|------|------|-------|
| WGS84 / UTM 33N | 32633 | Modern GPS-based |
| ED50 / UTM 33N | 23033 | Common in older Italian data |
| Monte Mario / Italy zone 2 | 3004 | Italian national grid (Gauss-Boaga) |

## Key Findings

### File Structure
- DWG files often use INSERT entities referencing block definitions rather than direct geometry
- Parcel boundaries are typically LWPOLYLINE entities, often not flagged as closed
- Layers are organized by municipality: `01_SERRA`, `02_CAPISTRANO`, `03_FABRIZIA`
- Label layers end in `_ETICHETTE` and contain CIRCLE/TEXT entities

### Polygonization Problems
1. **Open polylines** — boundaries drawn as line segments that visually close but aren't flagged closed
2. **Crossing lines without nodes** — lines pass through each other without intersection vertices
3. **Endpoint gaps** — small gaps (1-10m) between line endpoints due to drafting imprecision
4. **Missing segments** — some parcels have gaps of 100m+ requiring manual digitization

### Solutions
- **Noding:** `unary_union()` splits lines at all intersections
- **Endpoint snapping:** Cluster nearby endpoints (scipy hierarchical clustering) and merge
- **Polygonization:** `shapely.ops.polygonize()` finds closed regions from linework

## Scripts

### dxf_extract.py
Basic extraction of geometry from DXF files. Handles INSERT/block references.

```bash
python dxf_extract.py input.dxf -o output.geojson
# Options: --layer, --no-plot
```

### dxf_parcels.py
Advanced parcel extraction with polygonization, noding, and snapping.

```bash
python dxf_parcels.py input.dxf -o output_dir -l LAYER1 LAYER2 -c ed50-33n
# Options:
#   -l, --layers    Specific layers to process
#   -c, --crs       Source CRS: wgs84-33n, ed50-33n, monte-mario-2, wgs84-32n, ed50-32n
#   -t, --tolerance Snap tolerance in meters (default: 0.5)
#   -n, --max-plots Max individual parcel plots (default: 50)
#   --no-individual Skip per-parcel PNG generation
```

### dxf_diagnose.py
Diagnose linework issues — finds gaps, tests snap tolerances, visualizes problems.

```bash
python dxf_diagnose.py input.dxf -l LAYER_NAME -o output_dir
```

Outputs:
- `diagnose_LAYER.png` — 4-panel comparison of polygonization strategies
- `gaps_LAYER.png` — close-up views of isolated endpoints

### find_gaps.py
Find and visualize the largest gaps in linework, attempt to close them.

```bash
python find_gaps.py input.dxf -l LAYER_NAME -o output_dir -g 50
# -g, --max-gap   Maximum gap to close (meters)
```

### test_crs.py
Test different CRS assumptions to find correct datum.

```bash
# Test which CRS aligns with satellite imagery
python test_crs.py test input.dxf -l LAYER_NAME -o crs_test_output

# Apply manual offset to GeoJSON
python test_crs.py offset input.geojson -e -75 -n 50 -o corrected.geojson
```

## Typical Workflow

1. **Convert DWG to DXF** using ODA File Converter
   - Output format: "2018 ASCII DXF" works well

2. **Analyze structure:**
   ```bash
   python dxf_parcels.py input.dxf -o analysis --no-individual
   ```

3. **Diagnose problem layers:**
   ```bash
   python dxf_diagnose.py input.dxf -l PROBLEM_LAYER -o diag
   ```

4. **Try extraction with different CRSs, e.g.:**
   ```bash
   python dxf_parcels.py input.dxf -l GOOD_LAYERS -c ed50-33n -o parcels
   ```

5. **Hand-digitize missing parcels:**
   - See bosco/ep

## Output Format

GeoJSON FeatureCollection with properties:
```json
{
  "type": "Feature",
  "properties": {
    "layer": "01_SERRA",
    "parcel_index": 0,
    "area_deg2": 0.000123
  },
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[lon, lat], ...]]
  }
}
```
