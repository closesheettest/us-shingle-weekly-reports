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
    to_currency_numeric,  # for diagnostics & drilldowns
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

    # --- Session defaults for view persistence ---
    if "show_on_page" not in st.session_state:
        st.session_state["show_on_page"] = False
    if "rep_modal" not in st.session_state:
        st.session_state["rep_modal"] = None
    if "harv_modal" not in st.session_state:
        st.session_state["harv_modal"] = None

    # Hold computed artifacts in session so they persist across reruns
    # Keys we will use: rep_summary_df, company_df, harvester_df, harvester_pay_df, df_display, show_cols, rep_name_col

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

            # Buttons row
            c_show, c_hide = st.columns([1, 1])
            if c_show.button("👀 Show report on page"):
                # Read back from Excel to ensure exact match with download
                excel_bytes = io.BytesIO(wb_bytes)
                xls = pd.ExcelFile(excel_bytes, engine="openpyxl")
                rep_summary   = pd.read_excel(xls, sheet_name="Sales Rep Summary")
                company       = pd.read_excel(xls, sheet_name="Company Totals")
                harvester     = pd.read_excel(xls, sheet_name="Harvester Summary")
                harvester_pay = pd.read_excel(xls, sheet_name="Harvester Pay")

                # Build tagged DF for drilldowns (same logic as report)
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

                # Determine the sales rep column name in summary (first column fallback)
                rep_name_col = cols_map["sales_rep"]
                if rep_name_col not in rep_summary.columns:
                    rep_name_col = rep_summary.columns[0]

                # Store everything in session so it persists after rerun
                st.session_state["rep_summary_df"] = rep_summary
                st.session_state["company_df"] = company
                st.session_state["harvester_df"] = harvester
                st.session_state["harvester_pay_df"] = harvester_pay
                st.session_state["df_display"] = df_display
                st.session_state["show_cols"] = show_cols
                st.session_state["rep_name_col"] = rep_name_col

                st.session_state["show_on_page"] = True

            if c_hide.button("🙈 Hide on-page report"):
                st.session_state["show_on_page"] = False
                st.session_state["rep_modal"] = None
                st.session_state["harv_modal"] = None

            # ---- Render on-page report if flag is set ----
            if st.session_state.get("show_on_page", False):
                rep_summary   = st.session_state.get("rep_summary_df")
                company       = st.session_state.get("company_df")
                harvester     = st.session_state.get("harvester_df")
                harvester_pay = st.session_state.get("harvester_pay_df")
                df_display    = st.session_state.get("df_display")
                show_cols     = st.session_state.get("show_cols")
                rep_name_col  = st.session_state.get("rep_name_col")

                if any(x is None for x in [rep_summary, company, harvester, harvester_pay, df_display, show_cols, rep_name_col]):
                    st.warning("On-page data not ready. Click 'Show report on page' again.")
                else:
                    # Add "Action" column (visual only)
                    rep_table = rep_summary.copy()
                    if rep_name_col not in rep_table.columns:
                        rep_name_col = rep_table.columns[0]
                    rep_table["Action"] = "🔍 View"

                    harv_table = harvester.copy()
                    if "Harvester" not in harv_table.columns:
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
                        st.dataframe(_format_percent_columns(rep_table), use_container_width=True)
                        st.markdown("**Click a row’s button below to open details:**")
                        for i, row in rep_summary.reset_index(drop=True).iterrows():
                            rep_val = str(row[rep_name_col])
                            if st.button(f"🔍 View • {rep_val}", key=f"rep_view_{i}"):
                                st.session_state["rep_modal"] = rep_val

                        # Open/close modal
                        if st.session_state["rep_modal"] is not None:
                            sel_rep = st.session_state["rep_modal"]
                            with st.modal(f"Sales Rep Details — {sel_rep}"):
                                c1, c2, c3 = st.columns(3)
                                f_sit  = c1.checkbox("Only Sits (Sales logic)", key=f"rep_only_sits_modal_{sel_rep}")
                                f_sale = c2.checkbox("Only Sales", key=f"rep_only_sales_modal_{sel_rep}")
                                f_ns   = c3.checkbox("Only No-Shows", key=f"rep_only_noshow_modal_{sel_rep}")

                                mask = (df_display[rep_name_col].astype(str) == sel_rep)
                                if f_sit:
                                    mask &= df_display["is_sit_sales"]
                                if f_sale:
                                    mask &= df_display["is_sale"]
                                if f_ns:
                                    mask &= df_display["is_no_show"]

                                st.dataframe(df_display.loc[mask, show_cols].reset_index(drop=True), use_container_width=True)
                                st.caption("Close this popup to return to summaries.")
                                if st.button("Close", key=f"close_rep_modal_{sel_rep}"):
                                    st.session_state["rep_modal"] = None

                    # ---- COMPANY TOTALS ----
                    with t2:
                        st.dataframe(_format_percent_columns(company), use_container_width=True)

                    # ---- HARVESTER SUMMARY with per-row 'View' buttons + modal ----
                    with t3:
                        st.dataframe(_format_percent_columns(harv_table), use_container_width=True)
                        st.markdown("**Click a row’s button below to open details:**")
                        # Figure out harvester name column
                        harv_name_col = "Harvester" if "Harvester" in harvester.columns else harvester.columns[0]
                        for i, row in harvester.reset_index(drop=True).iterrows():
                            harv_val = str(row[harv_name_col])
                            if st.button(f"🔍 View • {harv_val}", key=f"harv_view_{i}"):
                                st.session_state["harv_modal"] = harv_val

                        if st.session_state["harv_modal"] is not None:
                            sel_harv = st.session_state["harv_modal"]
                            with st.modal(f"Harvester Details — {sel_harv}"):
                                d1, d2 = st.columns(2)
                                hf_sit  = d1.checkbox("Only Sits (Harvester logic)", key=f"harv_only_sits_modal_{sel_harv}")  # includes 'New Roof'
                                hf_sale = d2.checkbox("Only Sales", key=f"harv_only_sales_modal_{sel_harv}")

                                mask = (df_display["Harvester"].astype(str) == sel_harv)
                                if hf_sit:
                                    mask &= df_display["is_sit_harvester"]
                                if hf_sale:
                                    mask &= df_display["is_sale"]

                                st.dataframe(df_display.loc[mask, show_cols].reset_index(drop=True), use_container_width=True)
                                st.caption("Close this popup to return to summaries.")
                                if st.button("Close", key=f"close_harv_modal_{sel_harv}"):
                                    st.session_state["harv_modal"] = None

                    # ---- HARVESTER PAY (plain table) ----
                    with t4:
                        st.dataframe(harvester_pay, use_container_width=True)

            # -------- Diagnostics to troubleshoot mapping / statuses / currency --------
            with st.expander("🔎 Diagnostics (if numbers look wrong)"):
                try:
                    df_norm = normalize_columns(df)
                    st.write("**Columns in your file:**", list(df_norm.columns))

                    cols_resolved = ensure_columns(df_norm, cfg)
                    st.write("**Resolved column mapping (what the app will use):**")
                    st.json(cols_resolved)

                    # Status distribution
                    status_col = cols_resolved["status"]
                    status_values = (
                        df_norm[status_col].astype(str).str.strip()
                        .value_counts(dropna=False)
                        .reset_index()
                        .rename(columns={"index": "Status", status_col: "Count"})
                    )
                    st.write("**Status values found in your file (top 50):**")
                    st.dataframe(status_values.head(50), use_container_width=True)

                    # Tag & counts
                    df_tagged2 = tag_statuses(df_norm, status_col, cfg)
                    st.write("**Flag counts:**", {
                        "is_sit_sales sum": int(df_tagged2["is_sit_sales"].sum()),
                        "is_sit_harvester sum": int(df_tagged2["is_sit_harvester"].sum()),
                        "is_sale sum": int(df_tagged2["is_sale"].sum()),
                        "is_no_show sum": int(df_tagged2["is_no_show"].sum()),
                        "total rows": int(df_tagged2.shape[0]),
                    })

                    # Contract column check
                    amt_col2 = cols_resolved["total_contract"]
                    st.write("**Contract column sample (raw first 5):**", df_norm[amt_col2].head(5).tolist())
                    st.write("**Contract column sample (cleaned first 5):**", to_currency_numeric(df_norm[amt_col2]).head(5).tolist())

                    # Sum of sales $ (cleaned)
                    sales_amt_sum = float(to_currency_numeric(df_tagged2.loc[df_tagged2["is_sale"], amt_col2]).sum())
                    st.write("**Sum of $ for rows marked as Sales (cleaned):**", sales_amt_sum)

                    st.info("If flag counts are 0, fix the **Settings** tab (Sit/Sale/No-Show lists) to match the Status values above. If mapping is wrong, edit weekly_report_config.json to match your headers exactly.")
                except Exception as e:
                    st.error(f"Diagnostics failed: {e}")

            with st.expander("Preview uploaded data (first 100 rows)"):
                st.dataframe(df.head(100), use_container_width=True)

        except KeyError as e:
            st.error(f"Column mismatch: {e}")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Upload a file on the Reports tab to generate your report.")
