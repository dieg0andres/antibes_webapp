from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone


EXPENSE_SNAPSHOT_RE = re.compile(r"^ExpenseDB_\d{4}_\d{2}_\d{2}_\d{4}\.csv$")
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
TEXT_COLUMNS = [
    "Category",
    "Subcategory",
    "Merchant",
    "Description",
    "Source",
    "Comments",
]
PREVIEW_LIMIT = 10


def _stale_context(message: str) -> dict[str, Any]:
    return {
        "is_stale": True,
        "error_message": message,
        "row_counts": {"expenses": 0, "budget": 0},
        "source_files": {"expenses": None, "budget": None},
        "source_file_modified_at": {"expenses": None, "budget": None},
        "last_read_at": None,
        "expense_preview": [],
        "budget_preview": [],
        "metrics": {},
    }


def _get_cache_timeout():
    cache_ttl = getattr(settings, "PERSONAL_FINANCE_DASHBOARD_CACHE_TTL", None)
    if cache_ttl in (None, ""):
        return None

    try:
        return int(cache_ttl)
    except (TypeError, ValueError):
        return None


def _find_latest_expense_snapshot(data_dir: Path) -> Path | None:
    snapshots = sorted(
        (
            path
            for path in data_dir.glob("ExpenseDB_*.csv")
            if path.is_file() and EXPENSE_SNAPSHOT_RE.match(path.name)
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    return snapshots[0] if snapshots else None


def _validate_columns(df: pd.DataFrame, required_columns: list[str], label: str) -> str | None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        return f"{label} is missing required column(s): {', '.join(missing)}"
    return None


def _parse_money(series: pd.Series) -> pd.Series:
    cleaned = (
        series.fillna("")
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _normalize_expenses(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.loc[:, EXPENSE_COLUMNS].copy()
    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
    normalized["Amount"] = _parse_money(normalized["Amount"])

    for column in TEXT_COLUMNS:
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()

    return normalized


def _add_expense_metric_fields(df: pd.DataFrame) -> pd.DataFrame:
    metric_df = df.dropna(subset=["Date", "Amount"]).copy()
    metric_df["SpendAmount"] = metric_df["Amount"].abs()
    metric_df["Year"] = metric_df["Date"].dt.year.astype(int)
    metric_df["Month"] = metric_df["Date"].dt.strftime("%Y-%m")
    return metric_df


def _normalize_budget(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.loc[:, BUDGET_COLUMNS].copy()
    normalized["Monthly_Budget"] = _parse_money(normalized["Monthly_Budget"])
    normalized["Annual_Budget"] = _parse_money(normalized["Annual_Budget"])
    normalized["Category"] = normalized["Category"].fillna("").astype(str).str.strip()
    normalized["Subcategory"] = (
        normalized["Subcategory"].fillna("").astype(str).str.strip()
    )
    return normalized


def _serialize_expense_preview(df: pd.DataFrame) -> list[dict[str, Any]]:
    preview = df.head(PREVIEW_LIMIT).copy()
    preview["Date"] = preview["Date"].dt.strftime("%Y-%m-%d").fillna("")
    preview["Amount"] = preview["Amount"].fillna(0.0)
    return preview.to_dict(orient="records")


def _serialize_budget_preview(df: pd.DataFrame) -> list[dict[str, Any]]:
    preview = df.head(PREVIEW_LIMIT).copy()
    preview["Monthly_Budget"] = preview["Monthly_Budget"].fillna(0.0)
    preview["Annual_Budget"] = preview["Annual_Budget"].fillna(0.0)
    return preview.to_dict(orient="records")


def _round_amount(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return round(float(value), 2)


def _sum_records(
    df: pd.DataFrame,
    group_columns: list[str],
    *,
    amount_column: str = "SpendAmount",
    value_name: str = "amount",
    sort_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    if df.empty:
        return []

    grouped = (
        df.groupby(group_columns, dropna=False)[amount_column]
        .sum()
        .reset_index(name=value_name)
    )
    if sort_columns:
        grouped = grouped.sort_values(sort_columns)

    records = []
    for row in grouped.to_dict(orient="records"):
        record = dict(row)
        record[value_name] = _round_amount(record[value_name])
        records.append(record)
    return records


def _pareto_map(
    df: pd.DataFrame,
    key_columns: list[str],
    item_column: str,
    *,
    amount_column: str = "SpendAmount",
    value_name: str = "amount",
) -> dict[str, list[dict[str, Any]]]:
    if df.empty:
        return {}

    grouped = (
        df.groupby([*key_columns, item_column], dropna=False)[amount_column]
        .sum()
        .reset_index(name=value_name)
    )
    grouped = grouped.sort_values([*key_columns, value_name], ascending=[True] * len(key_columns) + [False])

    result: dict[str, list[dict[str, Any]]] = {}
    for key_values, rows in grouped.groupby(key_columns, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        key = "|".join(str(value) for value in key_values)
        result[key] = [
            {
                item_column.lower(): str(row[item_column]),
                value_name: _round_amount(row[value_name]),
            }
            for row in rows.to_dict(orient="records")
        ]
    return result


def _records_by_key(
    records: list[dict[str, Any]],
    key_column: str,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record[key_column])
        item = dict(record)
        item.pop(key_column, None)
        result.setdefault(key, []).append(item)
    return result


def _build_annual_metrics(expense_df: pd.DataFrame) -> dict[str, Any]:
    total_by_year = _sum_records(
        expense_df,
        ["Year"],
        sort_columns=["Year"],
    )
    category_stack_by_year = _sum_records(
        expense_df,
        ["Year", "Category"],
        sort_columns=["Year", "Category"],
    )
    subcategory_pareto = _pareto_map(
        expense_df,
        ["Year", "Category"],
        "Subcategory",
    )

    return {
        "total_by_year": total_by_year,
        "category_stack_by_year": category_stack_by_year,
        "subcategory_pareto_by_year_category": subcategory_pareto,
    }


def _build_monthly_metrics(expense_df: pd.DataFrame) -> dict[str, Any]:
    total_by_month = _sum_records(
        expense_df,
        ["Month"],
        sort_columns=["Month"],
    )
    category_by_month_records = _sum_records(
        expense_df,
        ["Category", "Month"],
        sort_columns=["Category", "Month"],
    )
    subcategory_stack_records = _sum_records(
        expense_df,
        ["Category", "Month", "Subcategory"],
        sort_columns=["Category", "Month", "Subcategory"],
    )
    merchant_pareto = _pareto_map(
        expense_df,
        ["Month", "Category", "Subcategory"],
        "Merchant",
    )
    subcategory_pareto_by_month = _pareto_map(
        expense_df,
        ["Month"],
        "Subcategory",
    )
    subcategory_pareto_by_month_category = _pareto_map(
        expense_df,
        ["Month", "Category"],
        "Subcategory",
    )

    return {
        "total_by_month": total_by_month,
        "category_by_month": _records_by_key(category_by_month_records, "Category"),
        "subcategory_stack_by_month_category": _records_by_key(
            subcategory_stack_records,
            "Category",
        ),
        "merchant_pareto_by_month_category_subcategory": merchant_pareto,
        "subcategory_pareto_by_month": subcategory_pareto_by_month,
        "subcategory_pareto_by_month_category": subcategory_pareto_by_month_category,
    }


def _budget_vs_actual_records(
    spend_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    group_columns: list[str],
) -> list[dict[str, Any]]:
    spend = (
        spend_df.groupby(group_columns, dropna=False)["SpendAmount"]
        .sum()
        .reset_index(name="spend_mtd")
    )
    budget = (
        budget_df.groupby(group_columns, dropna=False)["Monthly_Budget"]
        .sum()
        .reset_index(name="monthly_budget")
    )
    merged = budget.merge(spend, on=group_columns, how="outer")
    merged["monthly_budget"] = merged["monthly_budget"].fillna(0.0)
    merged["spend_mtd"] = merged["spend_mtd"].fillna(0.0)
    merged["remaining_budget"] = merged["monthly_budget"] - merged["spend_mtd"]
    merged["percent_used"] = merged.apply(
        lambda row: (
            row["spend_mtd"] / row["monthly_budget"] * 100
            if row["monthly_budget"]
            else 0.0
        ),
        axis=1,
    )
    merged = merged.sort_values(
        ["monthly_budget", *group_columns],
        ascending=[False] + [True] * len(group_columns),
    )

    records = []
    for row in merged.to_dict(orient="records"):
        record = {column: str(row[column]) for column in group_columns}
        record.update(
            {
                "spend_mtd": _round_amount(row["spend_mtd"]),
                "monthly_budget": _round_amount(row["monthly_budget"]),
                "remaining_budget": _round_amount(row["remaining_budget"]),
                "percent_used": _round_amount(row["percent_used"]),
            }
        )
        records.append(record)
    return records


def _budget_total_rows(budget_df: pd.DataFrame) -> pd.DataFrame:
    return budget_df.loc[
        budget_df["Category"].str.contains("total", case=False, na=False)
    ].copy()


def _grand_total_budget(budget_df: pd.DataFrame) -> float:
    total_rows = _budget_total_rows(budget_df)
    grand_total = total_rows.loc[
        total_rows["Category"].str.fullmatch("Grand Total", case=False, na=False)
    ]
    if grand_total.empty:
        return float(total_rows["Monthly_Budget"].sum(skipna=True))
    return float(grand_total["Monthly_Budget"].iloc[0])


def _category_budget_rows(budget_df: pd.DataFrame) -> pd.DataFrame:
    rows = _budget_total_rows(budget_df)
    rows = rows.loc[
        ~rows["Category"].str.fullmatch("Grand Total", case=False, na=False)
    ].copy()
    rows["Category"] = (
        rows["Category"]
        .str.replace(r"\s+Total$", "", regex=True, case=False)
        .str.strip()
    )
    return rows.loc[rows["Monthly_Budget"].fillna(0) > 0, ["Category", "Monthly_Budget"]]


def _subcategory_budget_rows(budget_df: pd.DataFrame) -> pd.DataFrame:
    rows = budget_df.loc[
        ~budget_df["Category"].str.contains("total", case=False, na=False)
    ].copy()
    rows = rows.loc[rows["Monthly_Budget"].fillna(0) > 0]
    return rows.loc[:, ["Category", "Subcategory", "Monthly_Budget"]]


def _build_current_month_budget_metrics(
    expense_df: pd.DataFrame,
    budget_df: pd.DataFrame,
) -> dict[str, Any]:
    if expense_df.empty:
        total_budget = _grand_total_budget(budget_df)
        return {
            "month": None,
            "total_spend_mtd": 0.0,
            "total_budget": _round_amount(total_budget),
            "category_budget_vs_actual": [],
            "subcategory_budget_vs_actual_by_category": {},
            "subcategory_budget_vs_actual": [],
        }

    current_month = str(expense_df["Month"].max())
    current_month_expenses = expense_df.loc[expense_df["Month"].eq(current_month)]
    total_budget = _grand_total_budget(budget_df)
    category_records = _budget_vs_actual_records(
        current_month_expenses,
        _category_budget_rows(budget_df),
        ["Category"],
    )
    subcategory_records = _budget_vs_actual_records(
        current_month_expenses,
        _subcategory_budget_rows(budget_df),
        ["Category", "Subcategory"],
    )
    subcategory_records = [
        record
        for record in subcategory_records
        if record["monthly_budget"] > 0
    ]

    return {
        "month": current_month,
        "total_spend_mtd": _round_amount(current_month_expenses["SpendAmount"].sum()),
        "total_budget": _round_amount(total_budget),
        "category_budget_vs_actual": category_records,
        "subcategory_budget_vs_actual_by_category": _records_by_key(
            subcategory_records,
            "Category",
        ),
        "subcategory_budget_vs_actual": subcategory_records,
    }


def _build_selector_metadata(expense_df: pd.DataFrame) -> dict[str, Any]:
    categories = sorted(category for category in expense_df["Category"].unique() if category)
    subcategories_by_category = {}
    for category, rows in expense_df.groupby("Category", dropna=False):
        if not category:
            continue
        subcategories_by_category[str(category)] = sorted(
            subcategory
            for subcategory in rows["Subcategory"].unique()
            if subcategory
        )

    return {
        "available_years": sorted(int(year) for year in expense_df["Year"].unique()),
        "available_months": sorted(str(month) for month in expense_df["Month"].unique()),
        "available_categories": categories,
        "available_subcategories_by_category": subcategories_by_category,
    }


def _build_metrics(
    expense_df: pd.DataFrame,
    budget_df: pd.DataFrame,
) -> dict[str, Any]:
    if expense_df.empty:
        return {
            "annual": {
                "total_by_year": [],
                "category_stack_by_year": [],
                "subcategory_pareto_by_year_category": {},
            },
            "monthly": {
                "total_by_month": [],
                "category_by_month": {},
                "subcategory_stack_by_month_category": {},
                "merchant_pareto_by_month_category_subcategory": {},
                "subcategory_pareto_by_month": {},
                "subcategory_pareto_by_month_category": {},
            },
            "current_month_budget": _build_current_month_budget_metrics(
                expense_df,
                budget_df,
            ),
            "available_years": [],
            "available_months": [],
            "available_categories": [],
            "available_subcategories_by_category": {},
        }

    return {
        "annual": _build_annual_metrics(expense_df),
        "monthly": _build_monthly_metrics(expense_df),
        "current_month_budget": _build_current_month_budget_metrics(
            expense_df,
            budget_df,
        ),
        **_build_selector_metadata(expense_df),
    }


def _format_mtime(path: Path) -> str:
    modified_at = timezone.datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.get_current_timezone())
    return modified_at.isoformat()


def _build_context_from_files() -> dict[str, Any]:
    data_dir = Path(getattr(settings, "PERSONAL_FINANCE_DATA_DIR"))
    expense_path = _find_latest_expense_snapshot(data_dir)
    budget_path = data_dir / "Budget.csv"

    if expense_path is None:
        return _stale_context(f"No ExpenseDB snapshot found in {data_dir}.")

    if not budget_path.is_file():
        return _stale_context(f"Budget file not found at {budget_path}.")

    try:
        expense_df = pd.read_csv(expense_path)
        budget_df = pd.read_csv(budget_path)
    except Exception as exc:
        return _stale_context(f"Personal finance CSV data could not be read: {exc}")

    expense_error = _validate_columns(expense_df, EXPENSE_COLUMNS, "ExpenseDB")
    if expense_error:
        return _stale_context(expense_error)

    budget_error = _validate_columns(budget_df, BUDGET_COLUMNS, "Budget")
    if budget_error:
        return _stale_context(budget_error)

    expense_df = _normalize_expenses(expense_df)
    budget_df = _normalize_budget(budget_df)
    metric_expense_df = _add_expense_metric_fields(expense_df)

    return {
        "is_stale": False,
        "error_message": None,
        "row_counts": {
            "expenses": int(len(expense_df)),
            "budget": int(len(budget_df)),
            "metric_expenses": int(len(metric_expense_df)),
            "excluded_from_metrics": int(len(expense_df) - len(metric_expense_df)),
        },
        "source_files": {
            "expenses": expense_path.name,
            "budget": budget_path.name,
        },
        "source_file_modified_at": {
            "expenses": _format_mtime(expense_path),
            "budget": _format_mtime(budget_path),
        },
        "last_read_at": timezone.localtime().isoformat(),
        "expense_preview": _serialize_expense_preview(expense_df),
        "budget_preview": _serialize_budget_preview(budget_df),
        "metrics": _build_metrics(metric_expense_df, budget_df),
    }


def build_personal_finance_dashboard_context(
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    cache_key = getattr(
        settings,
        "PERSONAL_FINANCE_DASHBOARD_CACHE_KEY",
        "personal_finance_dashboard_context",
    )

    if not force_refresh:
        cached_context = cache.get(cache_key)
        if isinstance(cached_context, dict):
            return cached_context

    context = _build_context_from_files()
    if not context.get("is_stale"):
        cache.set(cache_key, context, timeout=_get_cache_timeout())

    return context
