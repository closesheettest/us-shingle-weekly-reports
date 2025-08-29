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
    to_currency_numeric,
)

# ---------------- Page config ----------------
st.set_page_config(page_title="US Shingle Weekly Reports", page_icon="📊", layout="wide")

APP_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = APP_DIR / "weekly_report_config.json"
SETTINGS_PATH = APP_DIR / "settings.json"

# ---------------- Settings helpers ----------------
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
    cfg["_sit_statuses_lc"] = [s.lower() for s in cfg["sit_statuses"]]
    cfg["_sale_statuses_lc"] = [s.lower() for s in cfg["sale_statuses"]]
    cfg["_no_show_statuses_lc"] = [s.lower() for s in cfg["no_show_statuses"]]
    return cfg

# ---------------- Pretty % helper ----------------
def _format_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in list(out.columns):
        if isinstance(col, str) and col.endswith("%"):
            out[col] = (
                pd.to_numeric(out[col], errors="coerce").fillna(0.0) * 100
            ).round(0).astype("Int64").astype(str) + "%"
    return out

# ---------------- Init session ----------------
ss = st.session_state
ss.setdefault("data_ready", False)
ss.setdefault("show_on_page", False)
# stored artifacts (after upload):
# ss["wb_bytes"], ss["rep_summary_df"], ss["company_df"], ss["harvester_df"], ss["harvester_pay_df"]
# ss["df_tagged"], ss["cols_map"], ss["show_cols"]

# ---------------- UI ----------------
tabs = st.tabs(["📂 Reports", "🔧 Settings"])

with tabs[1]:
    st.header("Settings")
    st.caption("Define what counts as a Sit, Sale, and No Show (comma-separated).")
    current = get_settings()
    sit = st.text_input("Sit Statuses", value=", ".join(current.get("sit_statuses", [])))
    sale = st.text_input("Sale Statuses", value=", ".join(current.get("sale_statuses", [])))
    noshow = st.text_input("No Show Statuses", value=", ".join(current.get("no_show_statuses", [])))
    if st.button("Save Settings"):
        new_settings = save_settings(sit, sale, noshow)
        st.success("Saved!")
        st.json(new_settings)

with tabs[0]:
    st.title("📊 US Shingle Weekly Reports")
    st.caption("Upload once. Toggle **Show on-page report** to keep the report visible. Use the dropdowns to drill down by Sales Rep or Harvester.")

    # Upload
    uploaded_data = st.file_uploader(
        "Upload weekly data (.csv, .xlsx, .xlsm, .xls, .xlsb)",
        type=["csv", "xlsx", "xlsm", "xls", "xlsb"],
        key="uploader"
    )

    # HARVESTER toggle (new behavior)
    st.markdown("#### Harvester Counting Option")
    harv_count_resched = st.checkbox(
        'Count **"No Sit - rescheduled"** as a **Sit** in **Harvester** metrics (does **not** affect Sales metrics).',
        value=False,
        help="This only changes Harvester Sits/Sit%/Pay. Sales Sits/Sales remain the same."
    )

    # Process uploaded file
    if uploaded_data is not None:
        try:
            cfg = load_config_from_path(DEFAULT_CONFIG_PATH)
            cfg = merge_settings_into_config(cfg, get_settings())
            # inject the upload-time toggle into cfg so it affects both Excel and on-page views
            cfg["harvester_rescheduled_counts_as_sit"] = bool(harv_count_resched)

            df = read_input_file(uploaded_data)
            wb_bytes = build_workbook(df, cfg)

            # Tag dataframe for drilldowns (same logic as Excel)
            df_norm   = normalize_columns(df)
            cols_map  = ensure_columns(df_norm, cfg)
            df_tagged = tag_statuses(df_norm, cols_map["status"], cfg)
            harv_series = df_tagged[cols_map["appointment_set_by"]].fillna("").astype(str).str.strip().replace("", "Company")
            df_tagged["Harvester"] = harv_series
            df_tagged["$ Amount (clean)"] = to_currency_numeric(df_tagged[cols_map["total_contract"]])

            show_cols = [
                cols_map["job_name"],
                cols_map["date_created"],
                cols_map["start_date"],
                cols_map["status"],
                cols_map["sales_rep"],
                "Harvester",
                cols_map["total_contract"],
                "is_sit_sales",
                "is_sit_harvester",
                "is_sale",
                "is_no_show",
            ]

            # Read summary tables back from Excel (ensures match with download)
            xls = pd.ExcelFile(io.BytesIO(wb_bytes), engine="openpyxl")
            rep_summary   = pd.read_excel(xls, sheet_name="Sales Rep Summary")
            company       = pd.read_excel(xls, sheet_name="Company Totals")
            harvester     = pd.read_excel(xls, sheet_name="Harvester Summary")
            harvester_pay = pd.read_excel(xls, sheet_name="Harvester Pay")

            # Save to session
            ss["wb_bytes"] = wb_bytes
            ss["rep_summary_df"] = rep_summary
            ss["company_df"] = company
            ss["harvester_df"] = harvester
            ss["harvester_pay_df"] = harvester_pay
            ss["df_tagged"] = df_tagged
            ss["cols_map"] = cols_map
            ss["show_cols"] = show_cols
            ss["data_ready"] = True

            st.success("File processed. You can download or show the report on page.")

        except Exception as e:
            st.error(f"Error while processing file: {e}")

    # If we have data, show controls + content
    if ss["data_ready"]:
        st.download_button(
            "⬇️ Download Weekly_Reports.xlsx",
            data=ss["wb_bytes"],
            file_name="Weekly_Reports.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_btn"
        )

        # Persistent toggle to keep report visible
        ss["show_on_page"] = st.checkbox("👀 Show on-page report", value=ss["show_on_page"])

        if ss["show_on_page"]:
            rep_summary   = ss["rep_summary_df"]
            company       = ss["company_df"]
            harvester     = ss["harvester_df"]
            harvester_pay = ss["harvester_pay_df"]
            df_tagged     = ss["df_tagged"]
            cols_map      = ss["cols_map"]
            show_cols     = ss["show_cols"]

            t1, t2, t3, t4 = st.tabs([
                "Sales Rep Summary",
                "Company Totals",
                "Harvester Summary",
                "Harvester Pay",
            ])

            with t1:
                st.dataframe(_format_percent_columns(rep_summary), use_container_width=True)
                st.markdown("### 🔎 Drilldown by Sales Rep")
                reps = ["-- Select --"] + rep_summary.iloc[:, 0].astype(str).tolist()
                sel_rep = st.selectbox("Choose a Sales Rep", reps, key="sel_rep")
                c1, c2, c3 = st.columns(3)
                f_sit  = c1.checkbox("Only Sits (Sales logic)", key="rep_sits_filter")
                f_sale = c2.checkbox("Only Sales", key="rep_sales_filter")
                f_ns   = c3.checkbox("Only No-Shows", key="rep_noshow_filter")
                if sel_rep and sel_rep != "-- Select --":
                    mask = (df_tagged[cols_map["sales_rep"]].astype(str) == sel_rep)
                    if f_sit:  mask &= df_tagged["is_sit_sales"]
                    if f_sale: mask &= df_tagged["is_sale"]
                    if f_ns:   mask &= df_tagged["is_no_show"]
                    st.dataframe(df_tagged.loc[mask, show_cols].reset_index(drop=True), use_container_width=True)

            with t2:
                st.dataframe(_format_percent_columns(company), use_container_width=True)

            with t3:
                st.dataframe(_format_percent_columns(harvester), use_container_width=True)
                st.markdown("### 🔎 Drilldown by Harvester")
                harvesters = ["-- Select --"] + harvester.iloc[:, 0].astype(str).tolist()
                sel_harv = st.selectbox("Choose a Harvester", harvesters, key="sel_harv")
                d1, d2 = st.columns(2)
                hf_sit  = d1.checkbox("Only Sits (Harvester logic)", key="harv_sits_filter")  # includes New Roof and optionally Rescheduled
                hf_sale = d2.checkbox("Only Sales", key="harv_sales_filter")
                if sel_harv and sel_harv != "-- Select --":
                    mask = (df_tagged["Harvester"].astype(str) == sel_harv)
                    if hf_sit:  mask &= df_tagged["is_sit_harvester"]
                    if hf_sale: mask &= df_tagged["is_sale"]
                    st.dataframe(df_tagged.loc[mask, show_cols].reset_index(drop=True), use_container_width=True)

            with t4:
                st.dataframe(harvester_pay, use_container_width=True)
    else:
        st.info("Upload a file to generate your report (top of this tab).")
