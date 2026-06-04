/**
 * Section 1 (Griglie) map — parcel borders + sample-area markers for the
 * active grid.
 *
 * A thin specialization of ParcelMap: it supplies the marker styling/tooltip
 * and routes parcel/empty-space clicks to the writer's "new area here?" prompt.
 * All the map scaffolding (parcels, hover labels, fit / view persistence,
 * basemap sync, teardown) lives in ParcelMap.
 */
import { ParcelMap } from '../../base/js/parcel-map.js';
import * as S from '../../base/js/strings.js';

const MARKER_COLOR = '#2d5d2c';        // abies dark green
const MARKER_FILL_OPACITY = 0.85;

export class GriglieMap extends ParcelMap {
  /**
   * @param {object} opts
   * @param {HTMLElement} opts.container
   * @param {object} opts.geojson — terreni.geojson (sorted-by-area)
   * @param {string} [opts.basemap]
   * @param {function(object): void} [opts.onAreaClick] — area row clicked
   *   (typically opens a popover).
   * @param {function(number, number, object=): void} [opts.onEmptyClick] —
   *   (lat, lon, feature) when the map is clicked.  `feature` is the parcel
   *   polygon under the click (pre-fills the new-area form's region+parcel) or
   *   null off every parcel.  Writers use it to open the new-area prompt.
   * @param {{center:[number,number], zoom:number}} [opts.initialView]
   * @param {function([number,number], number): void} [opts.onViewChange]
   */
  constructor(opts) {
    super({
      container: opts.container,
      className: 'griglie-map',
      geojson: opts.geojson,
      basemap: opts.basemap,
      tools: { measure: true, location: true },
      initialView: opts.initialView,
      onViewChange: opts.onViewChange,
      onMapClick: opts.onEmptyClick
        ? (latlng, feature) => opts.onEmptyClick(latlng.lat, latlng.lng, feature)
        : null,
    });
    this.onAreaClick = opts.onAreaClick || null;
  }

  /**
   * Render markers for the active grid's sample areas.
   * @param {Array<{id, lat, lon, compresa, particella, numero}>} areas
   */
  setAreas(areas) {
    this._clearMarkers();
    for (const area of areas) {
      this._addAreaMarker(area, {
        fillColor: MARKER_COLOR,
        fillOpacity: MARKER_FILL_OPACITY,
        tooltip: this._areaTooltip(S.TOOLTIP_SAMPLE_AREA, area),
        onClick: (a) => { if (this.onAreaClick) this.onAreaClick(a); },
      });
    }
    this._refreshHighlight();
  }
}
