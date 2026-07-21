"""Update existing Parcel rows from canonical particelle.csv."""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.base import csv_io, csv_parcels
from apps.base.digests import mark_stale, regenerate_if_stale
from config.constants import DIGEST_PARCELS


class Command(BaseCommand):
    help = "Update existing Parcel rows from a canonical particelle.csv file."

    def add_arguments(self, parser):
        parser.add_argument("csv_file", type=Path)
        parser.add_argument(
            "--apply", action="store_true",
            help="Persist updates. Without this flag the command is a dry-run.",
        )

    def handle(self, *args, csv_file: Path, apply: bool, **opts):
        if not csv_file.is_file():
            raise CommandError(f"{csv_file} is not a file")
        parsed = _validated_rows(csv_file)
        updates, missing = csv_parcels.plan_existing_updates(parsed)
        if missing:
            raise CommandError(_format_missing(missing))
        if not apply:
            self.stdout.write(
                f"Would update {len(updates)} parcel row(s). "
                "Re-run with --apply to persist."
            )
            _write_update_sample(self.stdout, updates)
            return
        updated = csv_parcels.update_existing(parsed)
        mark_stale(DIGEST_PARCELS)
        regenerate_if_stale(DIGEST_PARCELS)
        self.stdout.write(f"Updated {updated} parcel row(s).")
        self.stdout.write("Regenerated parcels digest.")
        _write_update_sample(self.stdout, updates)


def _validated_rows(csv_file: Path):
    try:
        with csv_file.open('rb') as fh:
            reader = csv_io.read(
                fh, required_cols=csv_parcels.PARCEL_CSV_REQUIRED,
            )
    except csv_io.CsvError as exc:
        raise CommandError(str(exc)) from exc
    parsed, errors = csv_parcels.validate_rows(reader, csv_parcels.db_indexes())
    if errors:
        raise CommandError(_format_errors('Parcel CSV validation failed', errors))
    return parsed


def _format_errors(title: str, errors: list[str]) -> str:
    lines = [f"{title}:"]
    lines.extend(f"- {error}" for error in errors[:20])
    omitted = len(errors) - 20
    if omitted > 0:
        lines.append(f"- ... {omitted} more error(s) omitted")
    return "\n".join(lines)


def _format_missing(missing: list[tuple[str, str]]) -> str:
    labels = [f"{region}/{parcel}" for region, parcel in missing]
    return _format_errors('Parcel CSV contains rows not found in the DB', labels)


def _write_update_sample(stdout, updates):
    for update in updates[:20]:
        fields = ', '.join(update.fields)
        stdout.write(
            f"- {update.parcel.region.name}/{update.parcel.name}: {fields}"
        )
    omitted = len(updates) - 20
    if omitted > 0:
        stdout.write(f"- ... {omitted} more row(s) omitted")
