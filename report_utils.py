from __future__ import annotations
from pathlib import Path
import json
import io
from typing import Dict, List, Tuple

import pandas as pd


# ---------------------------
# File + Config helpers
# ---------------------------

def load_config_from_path(path: str | Path) -> dict:
    """Load a small JSON config (e.g., settings.json)."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_config_to_path(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def read_input_file(upload) -> pd.DataFrame:
    """Read CSV or Excel-like uploads to a DataFrame."""
    name = upload.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(upload)
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(upload)
    raise ValueError("Unsupported file type. Please upload .csv or .xlsx/.xlsm/.xls")


# ---------------------------
# Data normalization helpers
# ---------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip columns, replace spaces with underscores for easier access."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.lower()
    )
    return df


def coerce_money(series: pd.Series) -> pd.Series:
    """Convert money-like columns to numeric; strip commas/$."""
    return (
        pd.to_numeric(
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip(),
            errors="coerce",
        )
        .fillna(0.0)
    )


# ---------------------------
# Core metrics
# ---------------------------

def compute_sales_rep_table(
    df: pd.DataFrame,
    col_job: str,
    col_status: str,
    col_sales_rep: str,
    col_appt_by: str,
    col_insul: str,
    col_radiant: str,
    sale_statuses: List[str],
    sit_sales_statuses: List[str],
    sit_harvester_statuses: List[str],
    net_exclude_statuses: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (sales_rep_table, harvester_table).
    - NET excludes rows whose Status is in net_exclude_statuses from appointments, sits, and sales.
    - Harvester sits always count if Status in sit_harvester_statuses (even if excluded for NET).
    """
    # Standardize Status values for matching
    status_lc = df[col_status].astype(str).str.strip().str.lower()

    sale_set = {s.strip().lower() for s in sale_statuses}
    sit_sales_set = {s.strip().lower() for s in sit_sales_statuses}
    sit_harv_set = {s.strip().lower() for s in sit_harvester_statuses}
    net_excl_set = {s.strip().lower() for s in net_exclude_statuses}

    # Flags
    df["_is_sale"] = status_lc.isin(sale_set)
    df["_is_sit_sales"] = status_lc.isin(sit_sales_set)
    df["_is_sit_harv"] = status_lc.isin(sit_harv_set)
    df["_is_net_excl"] = status_lc.isin(net_excl_set)

    # Money
    df["_insul_amt"] = coerce_money(df[col_insul]) if col_insul in df else 0.0
    df["_rad_amt"] = coerce_money(df[col_radiant]) if col_radiant in df else 0.0

    # IRBAD "count into the sales" (count once if either sold)
    df["_irbad_flag"] = ((df["_insul_amt"] > 0) | (df["_rad_amt"] > 0)) & df["_is_sale"]

    # Group per Sales Rep
    groups = df.groupby(col_sales_rep, dropna=False)

    rows = []
    for rep, g in groups:
        rep = rep if pd.notna(rep) and str(rep).strip() != "" else "(Unassigned)"

        appointments = len(g)
        sits = int(g["_is_sit_sales"].sum())
        sales = int(g["_is_sale"].sum())

        # Exclusions for NET
        excl_appointments = int(g["_is_net_excl"].sum())
        excl_sits = int((g["_is_net_excl"] & g["_is_sit_sales"]).sum())
        excl_sales = int((g["_is_net_excl"] & g["_is_sale"]).sum())

        net_appts = max(appointments - excl_appointments, 0)
        net_sits = max(sits - excl_sits, 0)
        net_sales = max(sales - excl_sales, 0)

        sit_pct = (sits / appointments * 100.0) if appointments else 0.0
        net_sit_pct = (net_sits / net_appts * 100.0) if net_appts else 0.0
        sales_pct = (sales / sits * 100.0) if sits else 0.0
        net_sales_pct = (net_sales / net_sits * 100.0) if net_sits else 0.0

        insul_sum = float(g["_insul_amt"].sum())
        rad_sum = float(g["_rad_amt"].sum())
        irbad_sales_count = int(g["_irbad_flag"].sum())
        irbad_pct = (irbad_sales_count / sales * 100.0) if sales else 0.0

        rows.append(
            {
                "Sales Rep": rep,
                "Appointments": appointments,
                "Sits": sits,
                "Sit %": sit_pct,
                "Net Sits": net_sits,
                "Net Sit %": net_sit_pct,
                "Sales": sales,
                "Sales %": sales_pct,
                "Net Sales": net_sales,
                "Net Sales %": net_sales_pct,
                "Insulation $": insul_sum,
                "Radiant Barrier $": rad_sum,
                "IRBAD %": irbad_pct,
            }
        )

    sales_rep_table = pd.DataFrame(rows).sort_values(["Sales Rep"]).reset_index(drop=True)

    # Harvester report
    harv_df = df.copy()
    harv_df[col_appt_by] = harv_df[col_appt_by].fillna("").astype(str).str.strip()
    harv_df.loc[harv_df[col_appt_by] == "", col_appt_by] = "Company"

    h_groups = harv_df.groupby(col_appt_by, dropna=False)
    h_rows = []
    for harv, g in h_groups:
        appts = len(g)
        sits_h = int(g["_is_sit_harv"].sum())
        sit_pct_h = (sits_h / appts * 100.0) if appts else 0.0
        h_rows.append({"Harvester": harv, "Appointments": appts, "Sits": sits_h, "Sit %": sit_pct_h})
    harv_table = pd.DataFrame(h_rows).sort_values(["Harvester"]).reset_index(drop=True)

    return sales_rep_table, harv_table


# ---------------------------
# Optional: Excel export
# ---------------------------

def to_excel_bytes(tables: Dict[str, pd.DataFrame]) -> bytes:
    """Return an .xlsx workbook (in-memory) with each table on its own sheet."""
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for sheet_name, df in tables.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            # Very basic % formatting for known % columns
            workbook = writer.book
            pct_fmt = workbook.add_format({"num_format": "0%"})
            if not df.empty:
                for j, col in enumerate(df.columns):
                    if "%" in col:
                        # Apply to entire column (skip header)
                        writer.sheets[sheet_name].set_column(j, j, None, pct_fmt)
    return bio.getvalue()
