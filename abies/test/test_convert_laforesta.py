import csv

from ingest.convert_laforesta import (
    COL_CREW, COL_DATA, COL_EXTRA_NOTE, COL_HARVEST_PSR, COL_PARCEL,
    COL_PRODUCT, COL_PROT, COL_QUINTALS, COL_REGION, COL_VDP, OUT_HARVESTS,
    SRC_CREWS, _convert_harvests,
)


def _write_csv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
        writer.writerows(rows)


def _rows(path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter=','))


def test_harvest_converter_blanks_invalid_vdp_without_skipping_rows(tmp_path, capsys):
    src_dir = tmp_path / 'legacy'
    out_dir = tmp_path / 'canonical'
    src_dir.mkdir()
    out_dir.mkdir()

    header = [
        COL_REGION, COL_PARCEL, COL_DATA, COL_CREW, COL_PRODUCT,
        COL_QUINTALS, COL_VDP, COL_PROT, 'Note', COL_EXTRA_NOTE,
    ]
    _write_csv(src_dir / SRC_CREWS, header, [
        [
            'Capistrano', '1', '2026-01-02', 'Squadra A', 'Tronchi',
            '10', '783 bis', '1', '', 'bad vdp',
        ],
        [
            'Capistrano', '2', '2026-01-03', 'Squadra A', 'Tronchi',
            '11', '42', '2', 'psr', 'valid vdp',
        ],
    ])

    assert _convert_harvests(src_dir, out_dir) == 2

    rows = _rows(out_dir / OUT_HARVESTS)
    assert len(rows) == 2
    assert [r[COL_VDP] for r in rows] == ['', '42']
    assert [r[COL_PARCEL] for r in rows] == ['1', '2']
    assert rows[1][COL_HARVEST_PSR] == 'true'
    assert '1 VDP values blanked (non-integer; rows kept)' in capsys.readouterr().err
