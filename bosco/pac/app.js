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
        return fetch('../data/serra.geojson')
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
            /* Task 4 */
        },

        showDetails() {
            /* Task 5 */
        },

        exportCSV() {
            /* Task 5 */
        }
    };
})();

document.addEventListener('DOMContentLoaded', () => SamplingPlanner.init());
