from __future__ import annotations
import json
from pathlib import Path
import streamlit as st
import pandas as pd

from report_utils import (
    load_config_from_path,
    save_config_to_path,
    read_input_file,
    normalize_columns,
    compute_sales_rep_table,
    to_excel_bytes,
)

st.set_page_config(page_title="US Shingle Weekly Reports", layout="wide")


# ------------------------------------
# Session helpers
# ------------------------------------

SETTINGS_FILE = Path("settings.json")  # persisted mapping/status config


def get_settings() -> dict:
    cfg = load_config_from_path(SETTINGS_FILE)
    # Defaults
    cfg.setdefault("columns", {})
    # Column keys used by the app
    for key, default in {
        "job_name": "Job Name",
        "status": "Status",
        "sales_rep": "Sales Rep",
        "appointment_set_by": "Appointment Set By",
        "insulation_cost": "Insulation Cost",
        "radiant_barrier_cost": "Radiant Barrier Cost",
    }.items():
        cfg["columns"].setdefault(key, default)

    cfg.setdefault("sale_statuses", ["Signed Contract", "Sit - Sold"])
    cfg.setdefault("sit_sales_statuses", ["Signed Contract", "Sit - Sold", "Sit - Pending", "Sit - No Sale"])
    cfg.setdefault("sit_harvester_statuses", ["Signed Contract", "Sit - Sold", "Sit - Pending", "Sit - No Sale", "Credit Denial"])
    cfg.setdefault("net_exclude_statuses", ["Credit Denial"])
    return cfg


def save_settings(cfg: dict) -> None:
    save_config_to_path(cfg, SETTINGS_FILE)


# ------------------------------------
# UI
# ------------------------------------

st.title("US Shingle Weekly Reports")
st.caption("Single upload → map columns and statuses → Sales Rep & Harvester metrics.")

# One upload used everywhere
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xlsm", "xls"])
if uploaded:
    st.session_state["_raw_df"] = read_input_file(uploaded)
else:
    st.session_state.pop("_raw_df", None)

cfg = get_settings()

with st.expander("Settings", expanded=False):
    st.subheader("Column mapping")
    cols = cfg["columns"]
    # Offer automatic detection from uploaded file
    df_preview = None
    if "_raw_df" in st.session_state:
        df_preview = normalize_columns(st.session_state["_raw_df"])
        st.dataframe(df_preview.head(5), use_container_width=True)

    col_map = {}
    for label, key in [
        ("Job Name", "job_name"),
        ("Status", "status"),
        ("Sales Rep", "sales_rep"),
        ("Appointment Set By", "appointment_set_by"),
        ("Insulation Cost", "insulation_cost"),
        ("Radiant Barrier Cost", "radiant_barrier_cost"),
    ]:
        default = cols.get(key, label)
        col_map[key] = st.text_input(f"{label} column name", value=default)

    st.markdown("---")
    st.subheader("Status mapping")

    def tag_multiselect(label, current):
        return st.multiselect(label, options=sorted(_all_statuses(df_preview)), default=current)

    # Build all statuses seen in uploaded file (if any)
    def _all_statuses(df: pd.DataFrame | None):
        if df is None:
            return set(cfg.get("_last_statuses", ["Signed Contract", "Sit - Sold", "Credit Denial", "Sit - Pending", "Sit - No Sale"]))
        s_col = col_map["status"]
        if s_col.lower().replace(" ", "_") in df.columns:
            vals = df[s_col.lower().replace(" ", "_")].dropna().astype(str).str.strip().unique().tolist()
            cfg["_last_statuses"] = vals
            return set(vals)
        return set()

    sale_statuses = tag_multiselect("Sale (Sales) statuses", cfg["sale_statuses"])
    sit_sales = tag_multiselect("Sit (Sales) statuses", cfg["sit_sales_statuses"])
    sit_harv = tag_multiselect("Sit (Harvester) statuses", cfg["sit_harvester_statuses"])
    net_excl = tag_multiselect("NET exclude statuses (e.g., Credit Denial)", cfg["net_exclude_statuses"])

    if st.button("Save Settings"):
        cfg["columns"] = col_map
        cfg["sale_statuses"] = sale_statuses
        cfg["sit_sales_statuses"] = sit_sales
        cfg["sit_harvester_statuses"] = sit_harv
        cfg["net_exclude_statuses"] = net_excl
        save_settings(cfg)
        st.success("Settings saved.")

st.markdown("---")

# ------------------------------------
# Report
# ------------------------------------

if "_raw_df" not in st.session_state:
    st.info("Upload a file to generate reports.")
    st.stop()

raw_df = st.session_state["_raw_df"].copy()
df = normalize_columns(raw_df)

# Pull mapped columns (normalized)
c = cfg["columns"]
col_job = c["job_name"]
col_status = c["status"]
col_sales_rep = c["sales_rep"]
col_appt_by = c["appointment_set_by"]
col_insul = c["insulation_cost"]
col_rad = c["radiant_barrier_cost"]

# Normalize names to match df.columns (lower/underscores)
def norm(name: str) -> str:
    return name.strip().replace(" ", "_").lower()

col_job = norm(col_job)
col_status = norm(col_status)
col_sales_rep = norm(col_sales_rep)
col_appt_by = norm(col_appt_by)
col_insul = norm(col_insul)
col_rad = norm(col_rad)

missing = [x for x in [col_status, col_sales_rep, col_appt_by] if x not in df.columns]
if missing:
    st.error(f"Missing required columns in upload: {missing}. Check the column mapping in Settings.")
    st.stop()

sales_tbl, harv_tbl = compute_sales_rep_table(
    df=df,
    col_job=col_job,
    col_status=col_status,
    col_sales_rep=col_sales_rep,
    col_appt_by=col_appt_by,
    col_insul=col_insul if col_insul in df.columns else "",
    col_radiant=col_rad if col_rad in df.columns else "",
    sale_statuses=cfg["sale_statuses"],
    sit_sales_statuses=cfg["sit_sales_statuses"],
    sit_harvester_statuses=cfg["sit_harvester_statuses"],
    net_exclude_statuses=cfg["net_exclude_statuses"],
)

st.subheader("By Sales Rep")
st.dataframe(sales_tbl, use_container_width=True)

st.subheader("By Harvester")
st.dataframe(harv_tbl, use_container_width=True)

# KPIs (company totals)
if not sales_tbl.empty:
    total_row = {
        "Sales Rep": "Company Total",
        "Appointments": int(sales_tbl["Appointments"].sum()),
        "Sits": int(sales_tbl["Sits"].sum()),
        "Sit %": (sales_tbl["Sits"].sum() / sales_tbl["Appointments"].sum() * 100.0) if sales_tbl["Appointments"].sum() else 0.0,
        "Net Sits": int(sales_tbl["Net Sits"].sum()),
        "Net Sit %": (sales_tbl["Net Sits"].sum() / (sales_tbl["Appointments"].sum() - (sales_tbl["Appointments"].sum() - sales_tbl["Appointments"].sum())) * 100.0) if sales_tbl["Appointments"].sum() else 0.0,  # placeholder, already in table rows
        "Sales": int(sales_tbl["Sales"].sum()),
        "Sales %": (sales_tbl["Sales"].sum() / sales_tbl["Sits"].sum() * 100.0) if sales_tbl["Sits"].sum() else 0.0,
        "Net Sales": int(sales_tbl["Net Sales"].sum()),
        "Net Sales %": (sales_tbl["Net Sales"].sum() / sales_tbl["Net Sits"].sum() * 100.0) if sales_tbl["Net Sits"].sum() else 0.0,
        "Insulation $": float(sales_tbl["Insulation $"].sum()),
        "Radiant Barrier $": float(sales_tbl["Radiant Barrier $"].sum()),
        "IRBAD %": ( (sales_tbl["IRBAD %"] * sales_tbl["Sales"]).sum() / sales_tbl["Sales"].sum() ) if sales_tbl["Sales"].sum() else 0.0,
    }
    st.markdown("#### Company Totals")
    st.dataframe(pd.DataFrame([total_row]), use_container_width=True)

# Download workbook
tables = {
    "Sales by Rep": sales_tbl,
    "Harvesters": harv_tbl,
}
xlsx_bytes = to_excel_bytes(tables)
st.download_button(
    "Download Excel",
    data=xlsx_bytes,
    file_name="weekly_reports.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
