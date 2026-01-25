.PHONY: extract_polygons

extract_polygons: dxf_extract.py serra/Serra\ S.Bruno\ Particelle\ Forestali.dxf
    ./dxf_extract.py -o x -g 50 serra/Serra\ S.Bruno\ Particelle\ Forestali.dxf
    echo GeoJSON data extracted to x/all_parcels.geojson
