// Sampling Planner - Plan sample areas across forest parcels
const SamplingPlanner = (function() {
    'use strict';

    let mapWrapper = null;
    let leafletMap = null;
    let parcelLayer = null;
    let comprese = {};       // compresa name -> array of GeoJSON features
    let samplePoints = [];   // {lat, lng, parcel, adc} from last plan run
    let pointLayer = null;   // LayerGroup for sample point markers

    const PARCEL_STYLE = {
        color: '#3388ff',
        weight: 2,
        opacity: 0.8,
        fillOpacity: 0.1
    };

    function updateStatus(msg) {
        $('status').textContent = msg;
    }

    function loadParcels() {
        return fetch('../data/terreni.geojson')
            .then(r => r.json())
            .then(data => {
                const polygons = data.features.filter(
                    f => f.properties.type === 'polygon'
                );

                // Group by compresa (properties.layer)
                comprese = {};
                polygons.forEach(f => {
                    const name = f.properties.layer;
                    if (!comprese[name]) comprese[name] = [];
                    comprese[name].push(f);
                });

                // Display parcels on map with tooltips
                parcelLayer = L.geoJSON(
                    { type: 'FeatureCollection', features: polygons },
                    {
                        style: PARCEL_STYLE,
                        onEachFeature(feature, layer) {
                            layer.bindTooltip(feature.properties.name, { sticky: true });
                        }
                    }
                ).addTo(leafletMap);

                // Populate compresa dropdown
                const select = $('compresa-select');
                Object.keys(comprese).sort().forEach(name => {
                    const opt = document.createElement('option');
                    opt.value = name;
                    opt.textContent = `${name} (${comprese[name].length} particelle)`;
                    select.appendChild(opt);
                });

                return polygons.length;
            });
    }

    // --- Geometry helpers ---

    const DEG_TO_RAD = Math.PI / 180;

    function metersToDegLat(m) { return m / 111132.92; }
    function metersToDegLng(m, lat) { return m / (111132.92 * Math.cos(lat * DEG_TO_RAD)); }

    // Ray-casting point-in-ring test for a single ring (array of [lng, lat]).
    function pointInRing(lng, lat, ring) {
        let inside = false;
        for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
            const xi = ring[i][0], yi = ring[i][1];
            const xj = ring[j][0], yj = ring[j][1];
            if (((yi > lat) !== (yj > lat)) &&
                (lng < (xj - xi) * (lat - yi) / (yj - yi) + xi)) {
                inside = !inside;
            }
        }
        return inside;
    }

    // Point-in-polygon for GeoJSON Polygon geometry (exterior ring + holes).
    function pointInPolygon(lng, lat, geometry) {
        const coords = geometry.coordinates;
        if (!pointInRing(lng, lat, coords[0])) return false;
        for (let i = 1; i < coords.length; i++) {
            if (pointInRing(lng, lat, coords[i])) return false;
        }
        return true;
    }

    // Geodesic area of a GeoJSON Polygon feature in m².
    function featureArea(feature) {
        const ring = feature.geometry.coordinates[0];
        const latlngs = ring.map(c => ({ lat: c[1], lng: c[0] }));
        return MapCommon.geodesicArea(latlngs);
    }

    // Bounding box of an array of GeoJSON features.
    function boundingBox(features) {
        let minLng = Infinity, minLat = Infinity;
        let maxLng = -Infinity, maxLat = -Infinity;
        for (const f of features) {
            for (const c of f.geometry.coordinates[0]) {
                if (c[0] < minLng) minLng = c[0];
                if (c[0] > maxLng) maxLng = c[0];
                if (c[1] < minLat) minLat = c[1];
                if (c[1] > maxLat) maxLat = c[1];
            }
        }
        return { minLng, minLat, maxLng, maxLat };
    }

    // Return first feature containing the point, or null.
    function findContainingParcel(lng, lat, features) {
        for (const f of features) {
            if (pointInPolygon(lng, lat, f.geometry)) return f;
        }
        return null;
    }

    // Generate a regular grid of interior points, sorted south-to-north, west-to-east.
    function generateGrid(features, spacingLng, spacingLat) {
        const bb = boundingBox(features);
        const points = [];
        for (let lat = bb.minLat; lat <= bb.maxLat; lat += spacingLat) {
            for (let lng = bb.minLng; lng <= bb.maxLng; lng += spacingLng) {
                const parcel = findContainingParcel(lng, lat, features);
                if (parcel) {
                    points.push({ lat, lng, parcel: parcel.properties.name });
                }
            }
        }
        return points;
    }

    return {
        init() {
            mapWrapper = MapCommon.create('map', { basemap: 'satellite' });
            leafletMap = mapWrapper.getLeafletMap();

            pointLayer = L.layerGroup().addTo(leafletMap);

            (async () => {
                try {
                    const count = await loadParcels();

                    if (parcelLayer && parcelLayer.getBounds().isValid()) {
                        leafletMap.fitBounds(parcelLayer.getBounds());
                    }

                    updateStatus(`Caricate ${count} particelle`);
                } catch (err) {
                    updateStatus('Errore: ' + err.message);
                    console.error(err);
                }
            })();
        },

        setBasemap(name) {
            mapWrapper.setBasemap(name);
        },

        plan() {
            const compresaName = $('compresa-select').value;
            const features = comprese[compresaName];
            if (!features || features.length === 0) {
                updateStatus('Selezionare una compresa.');
                return;
            }

            const D = parseFloat($('diameter').value);
            const p = parseFloat($('coverage').value);
            if (!(D > 0)) {
                updateStatus('Diametro deve essere > 0.');
                return;
            }
            if (!(p > 0 && p <= 100)) {
                updateStatus('Copertura deve essere tra 0 e 100%.');
                return;
            }

            const totalAreaM2 = features.reduce((sum, f) => sum + featureArea(f), 0);
            const sampleAreaM2 = Math.PI / 4 * D * D;
            const targetN = Math.round((totalAreaM2 * p / 100) / sampleAreaM2);

            if (targetN < 1) {
                updateStatus('Parametri danno 0 punti.');
                return;
            }

            // Binary search on grid spacing to hit ~targetN interior points
            const bb = boundingBox(features);
            const midLat = (bb.minLat + bb.maxLat) / 2;
            let lo = 1;
            let hi = Math.sqrt(totalAreaM2);
            let bestPoints = [];

            for (let iter = 0; iter < 40; iter++) {
                const mid = (lo + hi) / 2;
                const spacingLat = metersToDegLat(mid);
                const spacingLng = metersToDegLng(mid, midLat);
                const pts = generateGrid(features, spacingLng, spacingLat);
                const count = pts.length;

                if (Math.abs(count - targetN) < Math.abs(bestPoints.length - targetN)) {
                    bestPoints = pts;
                }

                if (count === targetN) break;
                if (Math.abs(count - targetN) / targetN < 0.05) break;

                if (count > targetN) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }

            samplePoints = bestPoints.map((pt, i) => ({ ...pt, adc: i + 1 }));

            // Display points on map
            pointLayer.clearLayers();
            for (const pt of samplePoints) {
                const marker = L.circleMarker([pt.lat, pt.lng], {
                    radius: 5,
                    fillColor: '#ff4444',
                    color: '#000',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8
                });
                marker.bindTooltip(
                    '<b>ADC ' + pt.adc + '</b><br>Particella: ' + pt.parcel +
                    '<br>' + pt.lat.toFixed(6) + ', ' + pt.lng.toFixed(6)
                );
                marker.addTo(pointLayer);
            }

            // Show results (all values are internally computed, not user input)
            $('results-section').classList.remove('hidden');
            const statsEl = $('stats');
            statsEl.textContent = '';
            statsEl.appendChild(document.createTextNode(
                'Punti: ' + samplePoints.length + ' (obiettivo: ' + targetN + ')'
            ));
            statsEl.appendChild(document.createElement('br'));
            statsEl.appendChild(document.createTextNode(
                'Superficie compresa: ' + (totalAreaM2 / 10000).toFixed(2) + ' ha'
            ));
            statsEl.appendChild(document.createElement('br'));
            statsEl.appendChild(document.createTextNode(
                'Area singola ADC: ' + sampleAreaM2.toFixed(1) + ' m\u00B2'
            ));

            updateStatus('Pianificazione completata: ' + samplePoints.length + ' punti.');
        },

        showDetails() {
            const compresaName = $('compresa-select').value;
            const features = comprese[compresaName];
            if (!features || features.length === 0) return;

            const modal = $('details-modal');

            // Build per-parcel stats: count of sample points and area in ha
            const parcelStats = {};
            for (const f of features) {
                const name = f.properties.name;
                parcelStats[name] = { count: 0, areaHa: featureArea(f) / 10000 };
            }
            for (const pt of samplePoints) {
                if (parcelStats[pt.parcel]) parcelStats[pt.parcel].count++;
            }

            // Sort by parcel name using natural string sort
            const sorted = Object.keys(parcelStats).sort(
                (a, b) => a.localeCompare(b, undefined, { numeric: true })
            );

            // Extract short parcel name (part after the dash)
            function shortName(name) {
                const idx = name.indexOf('-');
                return idx >= 0 ? name.slice(idx + 1) : name;
            }

            // Build table
            let totalCount = 0;
            let totalAreaHa = 0;
            let html = '<table><thead><tr>' +
                '<th>Particella</th><th>N. ADC</th><th>ADC/ha</th>' +
                '</tr></thead><tbody>';
            for (const name of sorted) {
                const s = parcelStats[name];
                totalCount += s.count;
                totalAreaHa += s.areaHa;
                const density = s.areaHa > 0 ? (s.count / s.areaHa).toFixed(2) : '—';
                html += '<tr><td>' + shortName(name) + '</td>' +
                    '<td>' + s.count + '</td>' +
                    '<td>' + density + '</td></tr>';
            }
            const totalDensity = totalAreaHa > 0
                ? (totalCount / totalAreaHa).toFixed(2) : '—';
            html += '</tbody><tfoot><tr>' +
                '<td><b>Totale</b></td>' +
                '<td><b>' + totalCount + '</b></td>' +
                '<td><b>' + totalDensity + '</b></td>' +
                '</tr></tfoot></table>';

            $('details-title').textContent =
                'Aree di campionamento per ' + compresaName;
            $('details-body').innerHTML = html;

            // Set up close handlers once
            if (!modal._handlersSet) {
                $('details-close').addEventListener('click', e => {
                    e.preventDefault();
                    modal.classList.add('hidden');
                });
                modal.addEventListener('click', e => {
                    if (e.target === modal) modal.classList.add('hidden');
                });
                modal._handlersSet = true;
            }

            modal.classList.remove('hidden');
        },

        exportCSV() {
            if (samplePoints.length === 0) return;
            const compresaName = $('compresa-select').value;
            const lines = ['Compresa,Particella,ADC,Lng,Lat'];
            for (const pt of samplePoints) {
                lines.push(
                    compresaName + ',' + pt.parcel + ',' + pt.adc + ',' +
                    pt.lng.toFixed(6) + ',' + pt.lat.toFixed(6)
                );
            }
            const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'aree_di_campionamento.csv';
            a.click();
            URL.revokeObjectURL(a.href);
        }
    };
})();

document.addEventListener('DOMContentLoaded', () => SamplingPlanner.init());
