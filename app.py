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
    st.caption("Upload your weekly CSV/Excel. Download the Excel report or view it on this page. Click a rep/harvester to see the underlying rows.")

    uploaded_data = st.file_uploader(
        "Upload weekly data (.csv, .xlsx, .xlsm, .xls, .xlsb)",
        type=["csv", "xlsx", "xlsm", "xls", "xlsb"]
    )

    # ---- Session state for view persistence ----
    if "show_on_page" not in st.session_state:
        st.session_state["show_on_page"] = False
    if "active_rep" not in st.session_state:
        st.session_state["active_rep"] = None
    if "active_harv" not in st.session_state:
        st.session_state["active_harv"] = None

    # We’ll stash computed pieces here when you click "Show report on page"
    # rep_summary_df, company_df, harvester_df, harvester_pay_df
    # df_tagged, cols_map, show_cols

    if uploaded_data is not None:
        try:
            # Config + Settings
            cfg = load_config_from_path(DEFAULT_CONFIG_PATH)
            cfg = merge_settings_into_config(cfg, get_settings())

            # Read file
            df = read_input_file(uploaded_data)

            # Build downloadable Excel (the source of truth for numbers)
            wb_bytes = build_workbook(df, cfg)

            st.download_button(
                "⬇️ Download Weekly_Reports.xlsx",
                data=wb_bytes,
                file_name="Weekly_Reports.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Buttons to show/hide on-page view
            b1, b2 = st.columns(2)
            if b1.button("👀 Show report on page"):
                # Read the same four sheets back from the Excel bytes
                excel_bytes = io.BytesIO(wb_bytes)
                xls = pd.ExcelFile(excel_bytes, engine="openpyxl")
                rep_summary   = pd.read_excel(xls, sheet_name="Sales Rep Summary")
                company       = pd.read_excel(xls, sheet_name="Company Totals")
                harvester     = pd.read_excel(xls, sheet_name="Harvester Summary")
                harvester_pay = pd.read_excel(xls, sheet_name="Harvester Pay")

                # Build a tagged dataframe for drilldowns (same logic as report)
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

                # Stash in session so clicks don’t wipe the view
                st.session_state["rep_summary_df"] = rep_summary
                st.session_state["company_df"] = company
                st.session_state["harvester_df"] = harvester
                st.session_state["harvester_pay_df"] = harvester_pay
                st.session_state["df_tagged"] = df_tagged
                st.session_state["cols_map"] = cols_map
                st.session_state["show_cols"] = show_cols

                st.session_state["show_on_page"] = True

            if b2.button("🙈 Hide on-page report"):
                st.session_state["show_on_page"] = False
                st.session_state["active_rep"] = None
                st.session_state["active_harv"] = None

            # ---- Render on-page report if enabled ----
            if st.session_state["show_on_page"]:
                rep_summary   = st.session_state.get("rep_summary_df")
                company       = st.session_state.get("company_df")
                harvester     = st.session_state.get("harvester_df")
                harvester_pay = st.session_state.get("harvester_pay_df")
                df_tagged     = st.session_state.get("df_tagged")
                cols_map      = st.session_state.get("cols_map")
                show_cols     = st.session_state.get("show_cols")

                if any(x is None for x in [rep_summary, company, harvester, harvester_pay, df_tagged, cols_map, show_cols]):
                    st.warning("Data not ready. Click 'Show report on page' again.")
                else:
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
                        cols = st.columns(min(4, max(1, len(reps))))
                        for i, rep in enumerate(reps):
                            if cols[i % len(cols)].button(f"🔍 View {rep}", key=f"btn_rep_{i}"):
                                st.session_state["active_rep"] = rep

                        if st.session_state["active_rep"]:
                            sel_rep = st.session_state["active_rep"]
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
                                st.session_state["active_rep"] = None

                    # --- Company Totals ---
                    with t2:
                        st.dataframe(_format_percent_columns(company), use_container_width=True)

                    # --- Harvester Summary + inline drilldown ---
                    with t3:
                        st.dataframe(_format_percent_columns(harvester), use_container_width=True)
                        st.markdown("### 🔍 Drilldown by Harvester")
                        harvesters = harvester.iloc[:, 0].astype(str).tolist()
                        colsH = st.columns(min(4, max(1, len(harvesters))))
                        for i, harv in enumerate(harvesters):
                            if colsH[i % len(colsH)].button(f"🔍 View {harv}", key=f"btn_harv_{i}"):
                                st.session_state["active_harv"] = harv

                        if st.session_state["active_harv"]:
                            sel_h = st.session_state["active_harv"]
                            st.subheader(f"Details — {sel_h}")
                            d1, d2 = st.columns(2)
                            hf_sit  = d1.checkbox("Only Sits (Harvester logic)", key=f"harv_sits_{sel_h}")  # includes 'New Roof'
                            hf_sale = d2.checkbox("Only Sales", key=f"harv_sales_{sel_h}")

                            mask = (df_tagged["Harvester"].astype(str) == sel_h)
                            if hf_sit:  mask &= df_tagged["is_sit_harvester"]
                            if hf_sale: mask &= df_tagged["is_sale"]

                            st.dataframe(df_tagged.loc[mask, show_cols].reset_index(drop=True), use_container_width=True)
                            if st.button("Close details", key=f"close_harv_{sel_h}"):
                                st.session_state["active_harv"] = None

                    # --- Harvester Pay ---
                    with t4:
                        st.dataframe(harvester_pay, use_container_width=True)

            # ---- Diagnostics (optional) ----
            with st.expander("🔎 Diagnostics (if numbers look wrong)"):
                try:
                    df_norm = normalize_columns(df)
                    st.write("**Columns in your file:**", list(df_norm.columns))
                    cols_resolved = ensure_columns(df_norm, cfg)
                    st.write("**Resolved column mapping:**")
                    st.json(cols_resolved)

                    status_col = cols_resolved["status"]
                    status_values = (
                        df_norm[status_col].astype(str).str.strip()
                        .value_counts(dropna=False)
                        .reset_index()
                        .rename(columns={"index": "Status", status_col: "Count"})
                    )
                    st.write("**Status values found (top 50):**")
                    st.dataframe(status_values.head(50), use_container_width=True)

                    df_tagged2 = tag_statuses(df_norm, status_col, cfg)
                    st.write("**Flag counts:**", {
                        "is_sit_sales": int(df_tagged2["is_sit_sales"].sum()),
                        "is_sit_harvester": int(df_tagged2["is_sit_harvester"].sum()),
                        "is_sale": int(df_tagged2["is_sale"].sum()),
                        "is_no_show": int(df_tagged2["is_no_show"].sum()),
                        "rows": int(df_tagged2.shape[0]),
                    })

                    amt_col = cols_resolved["total_contract"]
                    st.write("**Contract column sample (raw):**", df_norm[amt_col].head(5).tolist())
                    st.write("**Contract column sample (cleaned):**", to_currency_numeric(df_norm[amt_col]).head(5).tolist())
                except Exception as e:
                    st.error(f"Diagnostics failed: {e}")

            with st.expander("Preview uploaded data (first 100 rows)"):
                st.dataframe(df.head(100), use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Upload a file to generate your report.")
