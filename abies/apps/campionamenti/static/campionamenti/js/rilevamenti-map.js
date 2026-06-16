/**
 * Section 2 (Rilevamenti) map — parcel borders + sample-area markers for the
 * active survey, coloured visited / unvisited from samples.json.
 *
 * A thin specialization of ParcelMap: it supplies the visited/unvisited marker
 * styling and clears the active area on a click off any marker.  All the map
 * scaffolding lives in ParcelMap.
 */
import { ParcelMap } from '../../base/js/parcel-map.js';
import * as S from '../../base/js/strings.js';

const VISITED_COLOR = '#2d5d2c';       // abies dark green
const UNVISITED_COLOR = '#8fbf8e';     // abies light green
const VISITED_FILL_OPACITY = 0.85;
const UNVISITED_FILL_OPACITY = 0.5;

export class RilevamentiMap extends ParcelMap {
  /**
   * @param {object} opts
   * @param {HTMLElement} opts.container
   * @param {object} opts.geojson — terreni.geojson (sorted-by-area)
   * @param {string} [opts.basemap]
   * @param {function(number|null): void} opts.onAreaSelect — sample_area_id, or
   *   null when the user clicks off every marker.
   * @param {{center:[number,number], zoom:number}} [opts.initialView]
   * @param {function([number,number], number): void} [opts.onViewChange]
   */
  constructor(opts) {
    super({
      container: opts.container,
      className: 'rilevamenti-map',
      geojson: opts.geojson,
      basemap: opts.basemap,
      tools: { measure: true, location: true },
      initialView: opts.initialView,
      onViewChange: opts.onViewChange,
      // A click off any marker (empty space or a parcel) clears the selection.
      onMapClick: () => {
        if (this.activeAreaId !== null) {
          this.setActiveAreaId(null);
          this.onAreaSelect(null);
        }
      },
    });
    this.onAreaSelect = opts.onAreaSelect;
  }

  /**
   * Render markers for the survey's sample areas.
   * @param {Array<{id, lat, lon, compresa, particella, numero}>} areas
   * @param {Map<number, {nAlberi:number}>} visitedById — missing = unvisited.
   */
  setAreas(areas, visitedById) {
    this._clearMarkers();
    for (const area of areas) {
      const visited = visitedById.get(area.id);
      const isVisited = !!visited;
      const tooltip = isVisited
        ? this._areaTooltip(S.TOOLTIP_SAMPLE_AREA_VISITED, area)
            .replace('{alberi}', S.SAMPLES_TREE_COUNT(visited.nAlberi))
        : this._areaTooltip(S.TOOLTIP_SAMPLE_AREA, area);
      this._addAreaMarker(area, {
        fillColor: isVisited ? VISITED_COLOR : UNVISITED_COLOR,
        fillOpacity: isVisited ? VISITED_FILL_OPACITY : UNVISITED_FILL_OPACITY,
        tooltip,
        onClick: (a) => this.onAreaSelect(a.id),
      });
    }
    this._refreshHighlight();
  }
}
