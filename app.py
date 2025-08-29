import streamlit as st
import json
from pathlib import Path
import pandas as pd

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
)

st.set_page_config(page_title="US Shingle Weekly Reports", page_icon="📊", layout="centered")

APP_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = APP_DIR / "weekly_report_config.json"
SETTINGS_PATH = APP_DIR / "settings.json"


# ---------- Settings (stored in a small JSON file) ----------
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
    # Update lists in cfg
    cfg["sit_statuses"] = settings.get("sit_statuses", cfg.get("sit_statuses", []))
    cfg["sale_statuses"] = settings.get("sale_statuses", cfg.get("sale_statuses", []))
    cfg["no_show_statuses"] = settings.get("no_show_statuses", cfg.get("no_show_statuses", []))
    # Keep the lowercase caches other functions expect
    cfg["_sit_statuses_lc"] = [s.lower() for s in cfg["sit_statuses"]]
    cfg["_sale_statuses_lc"] = [s.lower() for s in cfg["sale_statuses"]]
    cfg["_no_show_statuses_lc"] = [s.lower() for s in cfg["no_show_statuses"]]
    return cfg


# ---------- Helper for showing % nicely on page ----------
def _format_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Show any column whose name ends with % as whole-number percent strings like '75%'."""
    out = df.copy()
    for col in list(out.columns):
        if isinstance(col, str) and col.endswith("%"):
            out[col] = (
                pd.to_numeric(out[col], errors="coerce").fillna(0.0) * 100
            ).round(0).astype(int).astype(str) + "%"
    return out


# ---------- UI ----------
tabs = st.tabs(["📂 Reports", "⚙️ Settings"])

with tabs[1]:
    st.header("Settings")
    st.caption("Define what counts as a Sit, Sale, and No Show. Comma-separated.")
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
    st.caption("Upload your weekly CSV/Excel. You can download the Excel report or view it on this page.")

    uploaded_data = st.file_uploader(
        "Upload weekly data file (.csv, .xlsx, .xlsm, .xls, .xlsb)",
        type=["csv","xlsx","xlsm","xls","xlsb"]
    )

    if uploaded_data is not None:
        try:
            # Load config and merge your saved Settings
            cfg = load_config_from_path(DEFAULT_CONFIG_PATH)
            cfg = merge_settings_into_config(cfg, get_settings())

            # Read the input file and create the downloadable Excel
            df = read_input_file(uploaded_data)
            wb_bytes = build_workbook(df, cfg)

            st.success("Report generated successfully!")
            st.download_button(
                label="⬇️ Download Weekly_Reports.xlsx",
                data=wb_bytes,
                file_name="Weekly_Reports.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # ---- Build the same tables for on-page view ----
            df_norm   = normalize_columns(df)
            cols      = ensure_columns(df_norm, cfg)
            df_tagged = tag_statuses(df_norm, cols["status"], cfg)

            rep_summary   = compute_sales_rep_summary(df_tagged, cols, cfg["_truthy_values"])
            company       = compute_company_totals(df_tagged, cols)
            harvester     = compute_harvester_report(df_tagged, cols)
            harvester_pay = compute_harvester_pay(harvester)

            if st.button("👀 Show report on page"):
                t1, t2, t3, t4 = st.tabs([
                    "Sales Rep Summary",
                    "Company Totals",
                    "Harvester Summary",
                    "Harvester Pay",
                ])
                with t1:
                    st.dataframe(_format_percent_columns(rep_summary))
                with t2:
                    st.dataframe(_format_percent_columns(company))
                with t3:
                    st.dataframe(_format_percent_columns(harvester))
                with t4:
                    st.dataframe(harvester_pay)

            with st.expander("Preview your uploaded data (first 100 rows)"):
                st.dataframe(df.head(100))

        except KeyError as e:
            st.error(f"Column mismatch: {e}")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Upload a file on the Reports tab to generate your report.")
