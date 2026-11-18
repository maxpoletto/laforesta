/** Published roads for one compresa, independently switchable by type. */
export class BoscoRoads {
  constructor(map, features, panel, onToggle) {
    this.map = map;
    this.groups = new Map();
    const controls = panel.querySelector('[data-target="road-types"]');
    controls.replaceChildren();
    panel.hidden = !features.length;
    const renderer = L.svg();
    for (const feature of [...features].sort((a, b) =>
      a.properties.sort_order - b.properties.sort_order ||
      a.properties.road_type.localeCompare(b.properties.road_type))) {
      const {name, road_type, road_type_id} = feature.properties;
      if (!this.groups.has(road_type_id)) {
        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = true;
        checkbox.addEventListener('change', () => onToggle(road_type_id, checkbox.checked));
        label.append(checkbox, ' ' + road_type);
        controls.append(label);
        this.groups.set(road_type_id, {layer: L.featureGroup(), checkbox});
      }
      const tip = document.createElement('span');
      tip.textContent = name + ' · ' + road_type;
      L.geoJSON(feature, {
        style: {renderer, className: 'bosco-road', color: '#b55419', weight: 4},
      }).bindTooltip(tip, {sticky: true}).addTo(this.groups.get(road_type_id).layer);
    }
  }

  apply(hidden) {
    for (const [id, {layer, checkbox}] of this.groups) {
      checkbox.checked = !hidden.includes(id);
      if (checkbox.checked) layer.addTo(this.map);
      else layer.remove();
    }
  }
}
