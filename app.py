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
ss.setdefault("active_rep", None)
ss.setdefault("active_harv", None)

# These will be set when a file is processed:
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
    st.caption("Upload once. Then click **Show report on page**. Drilldowns work inline and persist across clicks.")

    # 1) Upload (only needed when you want to process a new file)
    uploaded_data = st.file_uploader(
        "Upload weekly data (.csv, .xlsx, .xlsm, .xls, .xlsb)",
        type=["csv", "xlsx", "xlsm", "xls", "xlsb"],
        key="uploader"
    )

    # 2) Process newly uploaded file (store everything in session so it persists)
    if uploaded_data is not None:
        try:
            cfg = load_config_from_path(DEFAULT_CONFIG_PATH)
            cfg = merge_settings_into_config(cfg, get_settings())

            # Read the file and build the Excel (source of truth)
            df = read_input_file(uploaded_data)
            wb_bytes = build_workbook(df, cfg)

            # Build tagged dataframe for drilldowns
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
                "is_sit_sales",      # sits for Sales reporting
                "is_sit_harvester",  # sits for Harvester reporting (includes 'New Roof')
                "is_sale",
                "is_no_show",
            ]

            # Read back summary tables from the generated Excel (matches the download)
            xls = pd.ExcelFile(io.BytesIO(wb_bytes), engine="openpyxl")
            rep_summary   = pd.read_excel(xls, sheet_name="Sales Rep Summary")
            company       = pd.read_excel(xls, sheet_name="Company Totals")
            harvester     = pd.read_excel(xls, sheet_name="Harvester Summary")
            harvester_pay = pd.read_excel(xls, sheet_name="Harvester Pay")

            # Save everything to session (so button clicks keep working)
            ss["wb_bytes"] = wb_bytes
            ss["rep_summary_df"] = rep_summary
            ss["company_df"] = company
            ss["harvester_df"] = harvester
            ss["harvester_pay_df"] = harvester_pay
            ss["df_tagged"] = df_tagged
            ss["cols_map"] = cols_map
            ss["show_cols"] = show_cols
            ss["data_ready"] = True
            ss["show_on_page"] = False  # require an explicit click each time you upload a new file
            ss["active_rep"] = None
            ss["active_harv"] = None

            st.success("File processed. You can download or show the report on page.")

        except Exception as e:
            st.error(f"Error while processing file: {e}")

    # 3) If we have data stored, show controls & content
    if ss["data_ready"]:
        # Download from session
        st.download_button(
            "⬇️ Download Weekly_Reports.xlsx",
            data=ss["wb_bytes"],
            file_name="Weekly_Reports.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_btn"
        )

        # Show/Hide on-page report toggles
        colA, colB = st.columns(2)
        if colA.button("👀 Show report on page", key="show_btn"):
            ss["show_on_page"] = True
        if colB.button("🙈 Hide on-page report", key="hide_btn"):
            ss["show_on_page"] = False
            ss["active_rep"] = None
            ss["active_harv"] = None

        # ---- Render on-page report (from session) ----
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

            # --- Sales Rep Summary + inline drilldown ---
            with t1:
                st.dataframe(_format_percent_columns(rep_summary), use_container_width=True)
                st.markdown("### 🔍 Drilldown by Sales Rep")
                reps = rep_summary.iloc[:, 0].astype(str).tolist()
                btn_cols = st.columns(min(4, max(1, len(reps))))
                for i, rep in enumerate(reps):
                    if btn_cols[i % len(btn_cols)].button(f"🔍 View {rep}", key=f"btn_rep_{i}"):
                        ss["active_rep"] = rep
                if ss["active_rep"]:
                    sel_rep = ss["active_rep"]
                    st.subheader(f"Details — {sel_rep}")
                    c1, c2, c3 = st.columns(3)
                    f_sit  = c1.checkbox("Only Sits (Sales logic)", key=f"rep_sits_{sel_rep}")
                    f_sale = c2.checkbox("Only Sales", key=f"rep_sales_{sel_rep}")
                    f_ns   = c3.checkbox("Only No-Shows", key=f"rep_noshow_{sel_rep}")

                    mask = (df_tagged[cols_map["sales_rep"]].astype(str) == sel_rep)
                    if f_sit:  mask &= df_tagged["is_sit_sales"]
                    if f_sale: mask &= df_tagged["is_sale"]
                    if f_ns:   mask &= df_tagged["is_no_show"]
                    st.dataframe(df_tagged.loc[mask, show_cols].reset_index(drop=True), use_container_width=True)
                    if st.button("Close details", key=f"close_rep_{sel_rep}"):
                        ss["active_rep"] = None

            # --- Company Totals ---
            with t2:
                st.dataframe(_format_percent_columns(company), use_container_width=True)

            # --- Harvester Summary + inline drilldown ---
            with t3:
                st.dataframe(_format_percent_columns(harvester), use_container_width=True)
                st.markdown("### 🔍 Drilldown by Harvester")
                harvesters = harvester.iloc[:, 0].astype(str).tolist()
                btn_cols_h = st.columns(min(4, max(1, len(harvesters))))
                for i, harv in enumerate(harvesters):
                    if btn_cols_h[i % len(btn_cols_h)].button(f"🔍 View {harv}", key=f"btn_harv_{i}"):
                        ss["active_harv"] = harv
                if ss["active_harv"]:
                    sel_h = ss["active_harv"]
                    st.subheader(f"Details — {sel_h}")
                    d1, d2 = st.columns(2)
                    hf_sit  = d1.checkbox("Only Sits (Harvester logic)", key=f"harv_sits_{sel_h}")  # includes 'New Roof'
                    hf_sale = d2.checkbox("Only Sales", key=f"harv_sales_{sel_h}")
                    mask = (df_tagged["Harvester"].astype(str) == sel_h)
                    if hf_sit:  mask &= df_tagged["is_sit_harvester"]
                    if hf_sale: mask &= df_tagged["is_sale"]
                    st.dataframe(df_tagged.loc[mask, show_cols].reset_index(drop=True), use_container_width=True)
                    if st.button("Close details", key=f"close_harv_{sel_h}"):
                        ss["active_harv"] = None

            # --- Harvester Pay ---
            with t4:
                st.dataframe(harvester_pay, use_container_width=True)

    else:
        st.info("Upload a file to generate your report (top of this tab).")
