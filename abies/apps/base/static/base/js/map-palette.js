/**
 * Basemap-aware colors for semantic map points.
 *
 * Green is legible on OSM/topo but disappears into satellite forest imagery.
 * Semantic dark/light-green points therefore become dark/pale yellow on the
 * satellite basemap. Categorical colors (species, parcel types, etc.) do not
 * use this helper: their color carries data and must remain stable.
 */

export const MARKER_TONE_DARK = 'dark';
export const MARKER_TONE_LIGHT = 'light';

export const SATELLITE_MARKER_COLORS = Object.freeze({
  [MARKER_TONE_DARK]: '#d6a800',
  [MARKER_TONE_LIGHT]: '#f0dda0',
});

const DEFAULT_MARKER_COLORS = Object.freeze({
  [MARKER_TONE_DARK]: '#2d5d2c',
  [MARKER_TONE_LIGHT]: '#8fbf8e',
});

function normalizedTone(tone) {
  return tone === MARKER_TONE_LIGHT ? MARKER_TONE_LIGHT : MARKER_TONE_DARK;
}

export function markerFillColor(basemap, tone, standardFillColor = null) {
  const normalized = normalizedTone(tone);
  if (basemap === 'satellite') return SATELLITE_MARKER_COLORS[normalized];
  return standardFillColor || DEFAULT_MARKER_COLORS[normalized];
}

/**
 * Add palette metadata to a Leaflet vector style and select its current fill.
 * Keeping the original fill in the marker options preserves each map's exact
 * existing green when switching back from satellite.
 */
export function semanticMarkerStyle(basemap, tone, style = {}) {
  const normalized = normalizedTone(tone);
  const standardFillColor = style.fillColor || DEFAULT_MARKER_COLORS[normalized];
  return {
    ...style,
    fillColor: markerFillColor(basemap, normalized, standardFillColor),
    abiesMarkerTone: normalized,
    abiesStandardFillColor: standardFillColor,
  };
}

/**
 * Recolor semantic vectors in a layer tree after a basemap change. Leaflet's
 * Canvas renderer coalesces the resulting setStyle calls into one redraw.
 */
export function refreshSemanticMarkers(layer, basemap) {
  if (!layer) return;
  const tone = layer.options?.abiesMarkerTone;
  if (tone && typeof layer.setStyle === 'function') {
    layer.setStyle({
      fillColor: markerFillColor(
        basemap, tone, layer.options.abiesStandardFillColor,
      ),
    });
  }
  if (typeof layer.eachLayer === 'function') {
    layer.eachLayer(child => refreshSemanticMarkers(child, basemap));
  }
}
