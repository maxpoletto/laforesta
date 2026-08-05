"""Shared dendrometric matrix exports for tree-like measurements."""

from collections import defaultdict
from decimal import Decimal

from apps.base import csv_io
from apps.base.digests import basal_area_m2, diameter_class_cm
from config import strings as S


def render_tree_dendrometry_csvs(trees) -> list[tuple[str, str]]:
    """Return tree-count, volume, and basal-area CSV matrices.

    Every matrix uses species on rows and a continuous sequence of 5 cm
    diameter classes on columns. Species whose values are all zero for a
    metric are omitted from that metric's CSV.
    """
    groups = defaultdict(lambda: {
        'tree_count': 0,
        'volume_m3': 0.0,
        'basal_area_m2': 0.0,
    })
    species_names = set()
    diameter_classes = set()

    for tree_row in trees:
        species = tree_row.tree.species.common_name
        diameter_class = diameter_class_cm(tree_row.d_cm)
        group = groups[(species, diameter_class)]
        group['tree_count'] += 1
        group['volume_m3'] += float(tree_row.volume_m3 or 0)
        group['basal_area_m2'] += basal_area_m2(tree_row.d_cm)
        species_names.add(species)
        diameter_classes.add(diameter_class)

    classes = _continuous_classes(diameter_classes)
    species = sorted(species_names, key=str.casefold)
    metrics = [
        (S.CSV_FILE_DENDROMETRY_TREE_COUNT, 'tree_count', 0),
        (S.CSV_FILE_DENDROMETRY_VOLUME, 'volume_m3', 4),
        (S.CSV_FILE_DENDROMETRY_BASAL_AREA, 'basal_area_m2', 4),
    ]
    return [
        (filename, _render_matrix_csv(groups, species, classes, metric, places))
        for filename, metric, places in metrics
    ]


def _continuous_classes(classes) -> list[int]:
    if not classes:
        return []
    return list(range(min(classes), max(classes) + 1, 5))


def _render_matrix_csv(groups, species, classes, metric, places) -> str:
    delimiter, decimal_sep = csv_io.export_format()
    buf, writer = csv_io.csv_buffer(delimiter)
    writer.writerow([S.COL_SPECIES, *classes])
    for species_name in species:
        values = [
            groups[(species_name, diameter_class)][metric]
            for diameter_class in classes
        ]
        if not any(values):
            continue
        if places:
            values = [
                csv_io.format_decimal(
                    Decimal(str(round(value, places))), decimal_sep,
                )
                for value in values
            ]
        writer.writerow([species_name, *values])
    return buf.getvalue()
