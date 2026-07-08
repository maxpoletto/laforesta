"""Replace sampled-tree domain rows from canonical sampled-trees.csv.

This is a one-off production repair path for the pre-1.0 duplicate sample
numbers that were removed from the canonical source CSV. It deliberately does
not recreate reference data, parcels, surveys, sample areas, marks, PAI, or
harvests.
"""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction

from apps.base import csv_containers as cc, csv_io
from apps.base.digests import mark_all_stale
from apps.base.models import (
    Sample, SampleArea, Species, Survey, TreeMark,
    TreePreserved, TreeSample,
)
from apps.campionamenti import csv_trees
from config import strings as S


class Command(BaseCommand):
    help = (
        "Replace all sampled-tree rows from canonical surveys.csv and "
        "sampled-trees.csv."
    )

    def add_arguments(self, parser):
        parser.add_argument("data_dir", type=Path)
        parser.add_argument(
            "--check", action="store_true",
            help="Validate and report only; persist nothing.",
        )

    def handle(self, *args, data_dir: Path, check: bool, **opts):
        if not data_dir.is_dir():
            raise CommandError(f"{data_dir} is not a directory")

        groups, survey_names = _validated_replacement_groups(data_dir)
        tree_ids = set(TreeSample.objects.values_list("tree_id", flat=True))
        blocked = _blocked_tree_ids(tree_ids)
        if blocked:
            sample = ", ".join(str(i) for i in sorted(blocked)[:20])
            raise CommandError(
                "Sampled-tree repair refused: sampled Tree rows are referenced "
                f"outside samples ({sample})."
            )

        n_samples = Sample.objects.count()
        n_tree_samples = TreeSample.objects.count()
        n_trees = len(tree_ids)
        n_new_trees = sum(len(parsed) for _survey, parsed in groups)

        with transaction.atomic():
            if not check:
                _delete_all_samples_and_trees(tree_ids)
                for survey, parsed in groups:
                    csv_trees.apply(survey, parsed)
                mark_all_stale()
            else:
                transaction.set_rollback(True)

        action = "Would replace" if check else "Replaced"
        self.stdout.write(
            f"{action} {n_samples} sample(s), {n_tree_samples} tree sample(s), "
            f"and {n_trees} sampled tree row(s) across "
            f"{len(survey_names)} survey/surveys; imported {n_new_trees} row(s)."
        )


def _validated_replacement_groups(data_dir: Path):
    survey_dates = _survey_dates(data_dir)
    reader = _read_required(
        data_dir / S.CSV_FILE_SAMPLED_TREES,
        [S.CSV_COL_SURVEY] + csv_trees.TREE_CSV_REQUIRED,
    )
    surveys = {s.name: s for s in Survey.objects.select_related("sample_grid")}
    has_date = S.CSV_COL_DATA in (reader.fieldnames or [])

    groups = []
    errors = []
    seen_surveys = set()
    for survey_name, rows in _group(reader, S.CSV_COL_SURVEY).items():
        survey = surveys.get(survey_name)
        if survey is None:
            errors.append(
                S.ERR_BOOTSTRAP_UNKNOWN_SURVEY.format(
                    S.CSV_FILE_SAMPLED_TREES, survey_name,
                )
            )
            continue
        sub = csv_io.CsvReader(rows, reader.delimiter, reader.fieldnames)
        parsed, group_errors = csv_trees.validate_rows(
            sub,
            _empty_sample_indexes(survey),
            has_date_column=has_date,
            default_date=survey_dates.get(survey_name),
        )
        errors.extend(group_errors)
        if parsed:
            groups.append((survey, parsed))
            seen_surveys.add(survey_name)
    if errors:
        raise CommandError(_format_errors(errors))
    return groups, seen_surveys


def _survey_dates(data_dir: Path) -> dict[str, object]:
    reader = _read_required(
        data_dir / S.CSV_FILE_SURVEYS,
        cc.SURVEY_CSV_REQUIRED,
    )
    parsed, errors = cc.validate_surveys(reader, cc.survey_db_indexes())
    existing = {
        s.name: s.sample_grid_id
        for s in Survey.objects.select_related("sample_grid")
    }
    dates = {}
    for row in parsed:
        survey = existing.get(row["name"])
        if survey is None:
            errors.append(
                S.ERR_BOOTSTRAP_UNKNOWN_SURVEY.format(
                    S.CSV_FILE_SURVEYS, row["name"],
                )
            )
            continue
        if survey != row["sample_grid"].id:
            errors.append(
                f"{S.CSV_FILE_SURVEYS}: {row['name']} uses a different grid "
                "than the existing survey."
            )
            continue
        dates[row["name"]] = row["date"]
    if errors:
        raise CommandError(_format_errors(errors))
    return dates


def _empty_sample_indexes(survey: Survey) -> csv_trees.TreeIndexes:
    return csv_trees.TreeIndexes(
        area_cache={
            (sa.parcel.region.name.lower(), sa.parcel.name, sa.number): sa
            for sa in SampleArea.objects
            .filter(sample_grid=survey.sample_grid)
            .select_related("parcel__region")
        },
        species_cache={s.common_name.lower(): s for s in Species.objects.all()},
        existing_sample_by_area={},
        existing_number_shoots=set(),
    )


def _blocked_tree_ids(tree_ids: set[int]) -> set[int]:
    if not tree_ids:
        return set()
    blocked = set(
        TreeMark.objects
        .filter(tree_id__in=tree_ids)
        .values_list("tree_id", flat=True)
    )
    blocked.update(
        TreePreserved.objects
        .filter(tree_id__in=tree_ids)
        .values_list("tree_id", flat=True)
    )
    return blocked


def _delete_all_samples_and_trees(tree_ids: set[int]) -> None:
    sample_ids = list(Sample.objects.values_list("id", flat=True))
    tree_sample_ids = list(TreeSample.objects.values_list("id", flat=True))
    if tree_sample_ids:
        _delete_by_ids("base_treesample", tree_sample_ids)
    if sample_ids:
        _delete_by_ids("base_sample", sample_ids)
    if tree_ids:
        _delete_by_ids("base_tree", sorted(tree_ids))


def _delete_by_ids(table: str, ids: list[int]) -> None:
    if not ids:
        return
    with connection.cursor() as cursor:
        cursor.executemany(f"DELETE FROM {table} WHERE id = %s", [(i,) for i in ids])


def _read_required(path: Path, required_cols: list[str]):
    if not path.is_file():
        raise CommandError(S.ERR_BOOTSTRAP_REQUIRED_FILE.format(path.name))
    try:
        return csv_io.read(path.read_text(encoding="utf-8-sig"),
                           required_cols=required_cols)
    except csv_io.CsvError as exc:
        raise CommandError(f"{path.name}: {exc}") from exc


def _group(reader, col):
    grouped = {}
    for row in reader:
        grouped.setdefault(row[col].strip(), []).append(row)
    return grouped


def _format_errors(errors: list[str]) -> str:
    return "\n".join(f"! {err}" for err in errors)
