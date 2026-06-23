import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convert_laforesta import (
    COL_ACC_M, COL_CREW, COL_DAMAGED, COL_DATA, COL_D_CM, COL_EXTRA_NOTE,
    COL_HARVEST_PSR, COL_H_MEASURED, COL_H_M, COL_LAT, COL_LON, COL_NUMBER,
    COL_OPERATOR, COL_PARCEL, COL_PRODUCT, COL_PROT, COL_QUINTALS, COL_REGION,
    COL_SPECIES, COL_VDP, OUT_HARVESTS, OUT_MARKS_DIR, SRC_CREWS,
    SRC_MARTELLATE_DIR, _convert_harvests, _convert_martellate,
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


def test_marks_converter_normalizes_upload_shape(tmp_path):
    src_dir = tmp_path / 'legacy'
    out_dir = tmp_path / 'canonical'
    martellate_dir = src_dir / SRC_MARTELLATE_DIR
    martellate_dir.mkdir(parents=True)
    out_dir.mkdir()

    _write_csv(
        martellate_dir / 'sample.csv',
        [
            COL_DATA, COL_REGION, COL_PARCEL, COL_DAMAGED, COL_NUMBER, 'Specie',
            COL_D_CM, COL_H_M, COL_H_MEASURED, COL_LAT, 'Lng', COL_ACC_M,
            COL_OPERATOR,
        ],
        [[
            '04/06/2026', 'Serra', '7', '1', '1969', 'Pino', '49', '31,5',
            '1', '38,559427', '16,286975', '2', 'Valerio',
        ]],
    )

    assert _convert_martellate(src_dir, out_dir) == 1

    rows = _rows(out_dir / OUT_MARKS_DIR / 'sample.csv')
    assert len(rows) == 1
    assert rows[0][COL_SPECIES] == 'Pino Laricio'
    assert rows[0][COL_H_M] == '31.5'
    assert rows[0][COL_LAT] == '38.559427'
    assert rows[0][COL_LON] == '16.286975'
    assert 'Specie' not in rows[0]
    assert 'Lng' not in rows[0]
