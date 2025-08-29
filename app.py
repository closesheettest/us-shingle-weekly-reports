import streamlit as st
import json
from pathlib import Path
import pandas as pd
import io

from report_utils import (
    load_config_from_path,
    read_input_file,
    build_workbook,
    normalize_columns,
    ensure_columns,
    tag_statuses,
    compute_sales_rep_summary,
    compute_company_totals,
    compute_harvester_report,
    compute_harvester_pay,
    to_currency_numeric,  # used in diagnostics & drilldowns
)

# ---------------- Page config ----------------
st.set_page_config(page_title="US Shingle Weekly Reports", page_icon="📊", layout="wide")

APP_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = APP_DIR / "weekly_report_config.json"
SETTINGS_PATH = APP_DIR / "settings.json"


# ---------------- Settings storage ----------------
def get_settings():
    defaults = {
        "sit_statuses": ["Sat", "Presented", "Met", "Demo"],
        "sale_statuses": ["Sold", "Approved", "Contract Signed"],
        "no_show_statuses": ["No Show", "Cancel", "Reschedule (No Sit)"]
    }
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text())
        except Exception:
            return defaults
    return defaults


def save_settings(sit, sale, noshow):
    data = {
        "sit_statuses": [s.strip() for s in sit.split(",") if s.strip()],
        "sale_statuses": [s.strip() for s in sale.split(",") if s.strip()],
        "no_show_statuses": [s.strip() for s in noshow.split(",") if s.strip()],
    }
    SETTINGS_PATH.write_text(json.dumps(data, indent=2))
    return data


def merge_settings_into_config(cfg, settings):
    cfg["sit_statuses"] = settings.get("sit_statuses", cfg.get("sit_statuses", []))
    cfg["sale_statuses"] = settings.get("sale_statuses", cfg.get("sale_statuses", []))
    cfg["no_show_statuses"] = settings.get("no_show_statuses", cfg.get("no_show_statuses", []))
    # keep lowercase caches used by report_utils
    cfg["_sit_statuses_lc"] = [s.lower() for s in cfg["sit_statuses"]]
    cfg["_sale_statuses_lc"] = [s.lower() for s in cfg["sale_statuses"]]
    cfg["_no_show_statuses_lc"] = [s.lower() for s in cfg["no_show_statuses"]]
    return cfg


# ---------------- Pretty % helper for in-page display ----------------
def _format_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in list(out.columns):
        if isinstance(col, str) and col.endswith("%"):
            out[col] = (
                pd.to_numeric(out[col], errors="coerce").fillna(0.0) * 100
            ).round(0).astype("Int64").astype(str) + "%"
    return out


# ---------------- UI ----------------
tabs = st.tabs(["📂 Reports", "🔧 Settings"])

with tabs[1]:
    st.header("Settings")
    st.caption("Define what counts as a Sit, Sale, and No Show. Use comma-separated values.")
    current = get_settings()
    sit = st.text_input("Sit Statuses", value=", ".join(current.get("sit_statuses", [])))
    sale = st.text_input("Sale Statuses", value=", ".join(current.get("sale_statuses", [])))
    noshow = st.text_input("No Show Statuses", value=", ".join(current.get("no_show_statuses", [])))
    if st.button("Save Settings"):
        new_settings = save_settings(sit, sale, noshow)
        st.success("Saved! These will be used for all future reports.")
        st.json(new_settings)

with tabs[0]:
    st.title("📊 US Shingle Weekly Reports")
    st.caption("Upload your weekly CSV/Excel. Download an Excel report or view it on this page. Click 🔍 View in the summaries to open a popup of the underlying rows.")

    uploaded_data = st.file_uploader(
        "Upload weekly data file (.csv, .xlsx, .xlsm, .xls, .xlsb)",
        type=["csv", "xlsx", "xlsm", "xls", "xlsb"]
    )

    if uploaded_data is not None:
        try:
            # Load config + merge Settings
            cfg = load_config_from_path(DEFAULT_CONFIG_PATH)
            cfg = merge_settings_into_config(cfg, get_settings())

            # Read uploaded file
            df = read_input_file(uploaded_data)

            # -------- Generate the Excel (download) --------
            wb_bytes = build_workbook(df, cfg)

            st.success("Report generated successfully!")
            st.download_button(
                label="⬇️ Download Weekly_Reports.xlsx",
                data=wb_bytes,
                file_name="Weekly_Reports.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Prepare a tagged dataframe for drilldowns/modals (same logic as report)
            df_norm   = normalize_columns(df)
            cols_map  = ensure_columns(df_norm, cfg)
            df_tagged = tag_statuses(df_norm, cols_map["status"], cfg)

            # Harvester column same as report (blank -> 'Company')
            harv_col = cols_map["appointment_set_by"]
            harv_series = df_tagged[harv_col].fillna("").astype(str).str.strip().replace("", "Company")

            amt_col = cols_map["total_contract"]
            df_display = df_tagged.copy()
            df_display["Harvester"] = harv_series
            df_display["$ Amount (clean)"] = to_currency_numeric(df_display[amt_col])

            # Visible columns in drilldown tables
            show_cols = [
                cols_map["job_name"],
                cols_map["date_created"],
                cols_map["start_date"],
                cols_map["status"],
                cols_map["sales_rep"],
                "Harvester",
                amt_col,
                "is_sit_sales",      # sits for Sales reporting
                "is_sit_harvester",  # sits for Harvester reporting (includes 'New Roof')
                "is_sale",
                "is_no_show",
            ]

            # -------- Show report on page (read back from Excel so it matches exactly) --------
            if st.button("👀 Show report on page"):
                try:
                    excel_bytes = io.BytesIO(wb_bytes)
                    xls = pd.ExcelFile(excel_bytes, engine="openpyxl")

                    rep_summary   = pd.read_excel(xls, sheet_name="Sales Rep Summary")
                    company       = pd.read_excel(xls, sheet_name="Company Totals")
                    harvester     = pd.read_excel(xls, sheet_name="Harvester Summary")
                    harvester_pay = pd.read_excel(xls, sheet_name="Harvester Pay")

                    # Add a '🔍 View' action column for Sales Rep Summary
                    rep_table = rep_summary.copy()
                    rep_name_col = cols_map["sales_rep"]  # this is the column title in the rep summary index reset
                    # Some files might label the first column differently; ensure we know which it is:
                    # rep_summary.reset_index(names=[rep_col]) in utils sets the first column to the rep col name.
                    # Use the first column name as fallback.
                    if rep_name_col not in rep_table.columns:
                        rep_name_col = rep_table.columns[0]

                    # Add a placeholder column for buttons (we render buttons separately in the loop)
                    rep_table["Action"] = "🔍 View"

                    # Add a '🔍 View' action column for Harvester Summary
                    harv_table = harvester.copy()
                    # Guarantee expected name
                    if "Harvester" not in harv_table.columns:
                        # first column fallback
                        harv_first = harv_table.columns[0]
                        harv_table = harv_table.rename(columns={harv_first: "Harvester"})
                    harv_table["Action"] = "🔍 View"

                    t1, t2, t3, t4 = st.tabs([
                        "Sales Rep Summary",
                        "Company Totals",
                        "Harvester Summary",
                        "Harvester Pay",
                    ])

                    # ---- SALES REP SUMMARY with per-row 'View' buttons + modal ----
                    with t1:
                        # We’ll show the table, then render buttons row-by-row underneath for click handling.
                        st.dataframe(_format_percent_columns(rep_table), use_container_width=True)

                        # Buttons for each row (unique keys)
                        st.markdown("**Click a row’s 🔍 View to open details:**")
                        for i, row in rep_summary.reset_index(drop=True).iterrows():
                            rep_val = str(row[rep_name_col])
                            # unique key per button
                            if st.button(f"🔍 View • {rep_val}", key=f"rep_view_{i}"):
