from __future__ import annotations

import csv
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable

import gspread
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone


EXPENSE_SNAPSHOT_RE = re.compile(r"^ExpenseDB_\d{4}_\d{2}_\d{2}_\d{4}\.csv$")
EXPENSE_SNAPSHOT_KEEP_COUNT = 7
EXPENSE_COLUMNS = [
    "Date",
    "Category",
    "Subcategory",
    "Merchant",
    "Description",
    "Amount",
    "Source",
    "Comments",
]
BUDGET_COLUMNS = ["Category", "Subcategory", "Monthly_Budget", "Annual_Budget"]


def _get_required_setting(name: str):
    value = getattr(settings, name, None)
    if value in (None, ""):
        raise CommandError(f"{name} must be configured.")
    return value


def _get_google_client(sa_key_path: Path) -> gspread.Client:
    if not sa_key_path.is_file():
        raise CommandError(f"Service account key not found at: {sa_key_path}")
    return gspread.service_account(filename=str(sa_key_path))


def _get_worksheet(spreadsheet: gspread.Spreadsheet, worksheet_gid: int):
    worksheet = spreadsheet.get_worksheet_by_id(int(worksheet_gid))
    if worksheet is None:
        raise CommandError(f"Worksheet gid not found: {worksheet_gid}")
    return worksheet


def _write_csv_atomic(path: Path, rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_name = tmp_file.name
            writer = csv.writer(tmp_file)
            writer.writerows(rows)

        os.replace(tmp_name, path)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _filter_rows_by_columns(
    rows: list[list[str]],
    columns: list[str],
) -> tuple[list[list[str]], list[str]]:
    if not rows:
        return [], columns

    headers = rows[0]
    missing_columns = [column for column in columns if column not in headers]
    if missing_columns:
        return [], missing_columns

    column_indexes = [headers.index(column) for column in columns]
    filtered_rows = [columns]
    for row in rows[1:]:
        filtered_rows.append(
            [row[index] if index < len(row) else "" for index in column_indexes]
        )

    return filtered_rows, []


def _cleanup_old_expense_snapshots(data_dir: Path) -> list[Path]:
    snapshots = sorted(
        (
            path
            for path in data_dir.glob("ExpenseDB_*.csv")
            if path.is_file() and EXPENSE_SNAPSHOT_RE.match(path.name)
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    old_snapshots = snapshots[EXPENSE_SNAPSHOT_KEEP_COUNT:]

    deleted = []
    for path in old_snapshots:
        path.unlink()
        deleted.append(path)

    return deleted


class Command(BaseCommand):
    help = "Export personal finance Google Sheets data to local CSV snapshots."

    def handle(self, *args, **options):
        spreadsheet_id = _get_required_setting("PERSONAL_FINANCE_SPREADSHEET_ID")
        sa_key_path = Path(_get_required_setting("PERSONAL_FINANCE_SA_KEY_PATH"))
        data_dir = Path(_get_required_setting("PERSONAL_FINANCE_DATA_DIR"))
        expense_gid = _get_required_setting("PERSONAL_FINANCE_EXPENSE_WORKSHEET_GID")
        budget_gid = _get_required_setting("BUDGET_SPREADSHEET_GID")

        now = timezone.localtime()
        expense_path = data_dir / f"ExpenseDB_{now.strftime('%Y_%m_%d_%H%M')}.csv"
        budget_path = data_dir / "Budget.csv"

        gc = _get_google_client(sa_key_path)
        spreadsheet = gc.open_by_key(spreadsheet_id)

        expense_worksheet = _get_worksheet(spreadsheet, expense_gid)
        budget_worksheet = _get_worksheet(spreadsheet, budget_gid)

        expense_rows, missing_expense_columns = _filter_rows_by_columns(
            expense_worksheet.get_all_values(),
            EXPENSE_COLUMNS,
        )
        if missing_expense_columns:
            self.stdout.write(
                self.style.WARNING(
                    "ExpenseDB export skipped. Missing required column(s): "
                    f"{', '.join(missing_expense_columns)}"
                )
            )
            return

        budget_rows, missing_budget_columns = _filter_rows_by_columns(
            budget_worksheet.get_all_values(),
            BUDGET_COLUMNS,
        )
        if missing_budget_columns:
            self.stdout.write(
                self.style.WARNING(
                    "Budget export skipped. Missing required column(s): "
                    f"{', '.join(missing_budget_columns)}"
                )
            )
            return

        _write_csv_atomic(expense_path, expense_rows)
        _write_csv_atomic(budget_path, budget_rows)
        deleted = _cleanup_old_expense_snapshots(data_dir)

        self.stdout.write(
            self.style.SUCCESS(
                "Personal finance data exported: "
                f"{expense_path}, {budget_path}; "
                f"deleted {len(deleted)} old ExpenseDB snapshot(s)."
            )
        )
