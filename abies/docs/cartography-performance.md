# Cartografia performance with a long Ipso walk

The benchmark uses one synthetic compresa with 144 contiguous parcels,
128 vertices per outline, and one Ipso working drawing with 25,000 fixes.
It moves one vertex on a shared parcel boundary and one vertex in the walk.
The data are generated in an isolated SQLite test database.

From the Abies directory with the development dependencies and Node installed,
run the opt-in benchmark.

```sh
python -m pytest -qs test/bench_cartography.py
```

## What the measurements cover

- Django's authenticated test client measures warm requests from the application
  through SQLite. Each median uses five requests after a warm-up. The client
  advertises gzip. The table records the actual Django response body and a
  separate size estimate for gzip at compression level 6.
- Node uses the editor's geometry edit function to retain ten undo snapshots.
  The heap delta is measured after garbage collection against the starting
  drawing. The timings cover taking a snapshot and moving a vertex.
- These measurements exclude the browser's map rendering, the production proxy
  and network latency.

## Results

One run on aarch64 with Python 3.14.4 and Node 22.22.1.

| Request | Median | JSON bytes | Django body bytes | Estimated gzip bytes |
| --- | ---: | ---: | ---: | ---: |
| Workspace GET without the walk | 2.9 ms | 8,352 | 8,352 | 839 |
| Workspace GET with the walk | 28.4 ms | 886,269 | 886,269 | 87,182 |
| Published map GeoJSON GET | 17.8 ms | 344,909 | 41,215 | 41,191 |
| Parcel boundary workspace GET | 47.7 ms | 23,834 | 23,834 | 2,828 |
| Save one shared boundary vertex POST | 50.4 ms | 112 | 112 | — |
| Save one drawing vertex POST | 123.9 ms | 2 | 2 | — |

The workspace reload after any successful edit includes every working
drawing in the compresa. This walk adds 877,917 bytes even when saving a
parcel boundary. Saving a drawing vertex also sends the whole drawing, a
777,835-byte request in this fixture. At a symmetric 1 Mbit/s with uncompressed
bodies, that upload and the following workspace reload would take about 13
seconds of transfer time before network latency or browser work.

The boundary endpoint indexes all 144 outlines and returns the selected
parcel and its neighbors. After a boundary save, the editor also reloads the
published GeoJSON, which covers the installation. The published map request
was already gzip-compressed in this run.

Moving one vertex in the long drawing took a median 17.1 ms for the edit and
snapshot alone. Holding all ten snapshots retained an additional 44.6 MiB
of JavaScript heap. The original drawing and rendered map layers are outside
that heap delta.
