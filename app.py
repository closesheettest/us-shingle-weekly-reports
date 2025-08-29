import streamlit as st
import json
from pathlib import Path
import pandas as pd
import io
from datetime import datetime

from report_utils import (
    load_config_from_path,
    read_input_file,
    build_workbook,
    normalize_columns,
    ensure_columns,
    tag_statuses,
    to_currency_numeric,
    compute_harvester_report,
    compute_harvester_pay,
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
    # global “rescheduled counts as sit” OFF; per-row overrides handle it
    cfg["harvester_rescheduled_counts_as_sit"] = False
    return cfg

# ---------------- Pretty % helper for on-page tables ----------------
def _format_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in list(out.columns):
        if isinstance(col, str) and col.endswith("%"):
            out[col] = (
                pd.to_numeric(out[col], errors="coerce").fillna(0.0) * 100
            ).round(0).astype("Int64").astype(str) + "%"
    return out

# ---------------- Row identity for overrides ----------------
def make_override_id(row, cols_map):
    # Stable identity using fields in the sheet
    j = str(row[cols_map["job_name"]])
    sd = str(row[cols_map["start_date"]])
    dc = str(row[cols_map["date_created"]])
    return f"{j} | {sd} | {dc}"

# ---------------- Apply per-row overrides (Harvester sits only) ----------------
def apply_harvester_overrides(df_tagged: pd.DataFrame, override_ids: set) -> pd.DataFrame:
    if not override_ids:
        return df_tagged
    df2 = df_tagged.copy()
    df2.loc[df2["override_id"].isin(override_ids), "is_sit_harvester"] = True
    return df2

# ---------------- Build print-friendly HTML ----------------
def build_print_html(
    company_df: pd.DataFrame,
    rep_df: pd.DataFrame,
    harv_df: pd.DataFrame,
    pay_df: pd.DataFrame,
    overrides_table: pd.DataFrame,
    report_title: str = "US Shingle Weekly Report"
) -> str:
    # Simple, clean HTML with print styles
    style = """
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; }
      h1 { margin: 0 0 8px 0; }
      h2 { margin: 24px 0 8px 0; }
      .meta { color:#555; margin-bottom: 16px; }
      table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }
      th, td { border: 1px solid #ddd; padding: 8px; }
      th { background: #f3f3f3; text-align: left; }
      .note { font-size: 12px; color: #444; }
      @media print { .no-print { display: none; } }
    </style>
    """
    def df_to_html(df):
        # keep as-is (assumes safe input)
        return df.to_html(index=False, border=0)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = [style, f"<h1>{report_title}</h1>", f"<div class='meta'>Generated: {now}</div>"]
    html += ["<h2>Company Totals</h2>", df_to_html(company_df)]
    html += ["<h2>Sales Rep Summary</h2>", df_to_html(rep_df)]
    html += ["<h2>Harvester Summary (with overrides)</h2>", df_to_html(harv_df)]
    html += ["<h2>Harvester Pay (with overrides)</h2>", df_to_html(pay_df)]

    if not overrides_table.empty:
        html += ["<h2>Override Notes</h2>", df_to_html(overrides_table)]
        html += ["<div class='note'>Only rows listed above were manually overridden to count as Harvester sits. Sales metrics were not affected.</div>"]
    else:
        html += ["<h2>Override Notes</h2>", "<p>No overrides applied.</p>"]

    return "\n".join(html)

# ---------------- Init session ----------------
ss = st.session_state
ss.setdefault("data_ready", False)
ss.setdefault("show_on_page", False)
# per-row harvester overrides (set of override_id strings)
ss.setdefault("harv_overrides", set())
# notes for overrides: map override_id -> note string
ss.setdefault("harv_override_notes", {})

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
    st.caption("Upload once. Toggle **Show on-page report** to keep it visible. In the Harvester tab you can override ANY row to count as a sit (Harvester-only) and MUST provide a note for each override. Sales metrics are unaffected.")

    # Upload
    uploaded_data = st.file_uploader(
        "Upload weekly data (.csv, .xlsx, .xlsm, .xls, .xlsb)",
        type=["csv", "xlsx", "xlsm", "xls", "xlsb"],
        key="uploader"
    )

    # Process the uploaded file
    if uploaded_data is not None:
        try:
            cfg = load_config_from_path(DEFAULT_CONFIG_PATH)
            cfg = merge_settings_into_config(cfg, get_settings())

            df = read_input_file(uploaded_data)
            wb_bytes = build_workbook(df, cfg)  # official download (no per-row overrides)

            # Tag for drilldowns (same base rules as workbook)
            df_norm   = normalize_columns(df)
            cols_map  = ensure_columns(df_norm, cfg)
            df_tagged = tag_statuses(df_norm, cols_map["status"], cfg)

            # Precompute convenience columns
            df_tagged["Harvester"] = (
                df_tagged[cols_map["appointment_set_by"]].fillna("").astype(str).str.strip().replace("", "Company")
            )
            df_tagged["$ Amount (clean)"] = to_currency_numeric(df_tagged[cols_map["total_contract"]])
            df_tagged["_status_lc"] = df_tagged[cols_map["status"]].astype(str).str.strip().str.lower()
            df_tagged["override_id"] = df_tagged.apply(lambda r: make_override_id(r, cols_map), axis=1)

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

            # Read summaries from the generated Excel (guarantees match with download)
            xls = pd.ExcelFile(io.BytesIO(wb_bytes), engine="openpyxl")
            rep_summary   = pd.read_excel(xls, sheet_name="Sales Rep Summary")
            company       = pd.read_excel(xls, sheet_name="Company Totals")
            harvester     = pd.read_excel(xls, sheet_name="Harvester Summary")
            harvester_pay = pd.read_excel(xls, sheet_name="Harvester Pay")

            # Save to session
            ss["wb_bytes"] = wb_bytes
            ss["rep_summary_df"] = rep_summary
            ss["company_df"] = company
            ss["harvester_df"] = harvester      # base (no overrides)
            ss["harvester_pay_df"] = harvester_pay  # base (no overrides)
            ss["df_tagged"] = df_tagged
            ss["cols_map"] = cols_map
            ss["show_cols"] = show_cols
            ss["data_ready"] = True
            # Reset overrides/notes when a new file is uploaded
            ss["harv_overrides"] = set()
            ss["harv_override_notes"] = {}

            st.success("File processed. You can download or show the report on page.")

        except Exception as e:
            st.error(f"Error: {e}")

    # If we have data, show controls + content
    if ss["data_ready"]:
        # Official unmodified workbook download
        st.download_button(
            "⬇️ Download Weekly_Reports.xlsx (no overrides)",
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
            df_tagged     = ss["df_tagged"]
            cols_map      = ss["cols_map"]
            show_cols     = ss["show_cols"]

            # EFFECTIVE DF using overrides with notes (only apply overrides that have a note)
            overrides_all = set(ss.get("harv_overrides", set()))
            notes_map = ss.get("harv_override_notes", {})
            valid_overrides = {oid for oid in overrides_all if notes_map.get(oid, "").strip()}
            if overrides_all and (len(valid_overrides) < len(overrides_all)):
                st.warning("Overrides without notes are not applied. Please add a note for each selected override.", icon="⚠️")

            df_effective = apply_harvester_overrides(df_tagged, valid_overrides)
            harv_summary_effective = compute_harvester_report(df_effective, cols_map)
            harv_pay_effective = compute_harvester_pay(harv_summary_effective)

            # Build "overrides with notes" table for print/export
            if valid_overrides:
                # pick some identifying columns to show
                over_mask = df_tagged["override_id"].isin(valid_overrides)
                overrides_table = df_tagged.loc[
                    over_mask,
                    [cols_map["job_name"], cols_map["start_date"], cols_map["date_created"], "Harvester", "override_id"]
                ].copy()
                overrides_table["Note"] = overrides_table["override_id"].map(lambda oid: notes_map.get(oid, ""))
                overrides_table = overrides_table.rename(columns={
                    cols_map["job_name"]: "Job Name",
                    cols_map["start_date"]: "Start Date",
                    cols_map["date_created"]: "Date Created",
                })
            else:
                overrides_table = pd.DataFrame(columns=["Job Name","Start Date","Date Created","Harvester","override_id","Note"])

            # Add override indicator columns for drilldowns
            def inject_override_cols(df_in: pd.DataFrame) -> pd.DataFrame:
                out = df_in.copy()
                out["Override Applied (Harvester Sit)"] = out["override_id"].isin(valid_overrides)
                out["Override Note"] = out["override_id"].map(lambda oid: notes_map.get(oid, ""))
                return out

            t1, t2, t3, t4, t5 = st.tabs([
                "Sales Rep Summary",
                "Company Totals",
                "Harvester Summary (with overrides)",
                "Harvester Pay (with overrides)",
                "🖨️ Print / Export"
            ])

            # --- Sales Rep Summary (unchanged by overrides) ---
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
                    mask = (df_effective[cols_map["sales_rep"]].astype(str) == sel_rep)
                    if f_sit:  mask &= df_effective["is_sit_sales"]
                    if f_sale: mask &= df_effective["is_sale"]
                    if f_ns:   mask &= df_effective["is_no_show"]
                    drill = inject_override_cols(df_effective.loc[mask, show_cols + ["override_id"]])
                    st.dataframe(drill.drop(columns=["override_id"]).reset_index(drop=True), use_container_width=True)

            # --- Company Totals (Sales-based; unaffected by overrides) ---
            with t2:
                st.dataframe(_format_percent_columns(company), use_container_width=True)

            # --- Harvester Summary WITH per-row overrides + drilldown & picker ---
            with t3:
                st.info("This table reflects **your per-row overrides** (each override must include a Note).")
                st.dataframe(_format_percent_columns(harv_summary_effective), use_container_width=True)

                st.markdown("### 🔧 Overrides: Mark ANY rows as Harvester Sits (Note required)")
                # Choose a harvester to manage rows
                harvesters = ["-- Select --"] + harv_summary_effective["Harvester"].astype(str).tolist()
                sel_harv = st.selectbox("Choose a Harvester to manage overrides", harvesters, key="ov_sel_harv")

                if sel_harv and sel_harv != "-- Select --":
                    cols_for_labels = [
                        cols_map["job_name"], cols_map["start_date"], cols_map["date_created"], "override_id"
                    ]
                    subset = df_tagged.loc[df_tagged["Harvester"].astype(str) == sel_harv, cols_for_labels].copy()

                    if subset.empty:
                        st.caption("No rows for this harvester.")
                    else:
                        subset["Label"] = (
                            subset[cols_map["job_name"]].astype(str)
                            + " — Start: " + subset[cols_map["start_date"]].astype(str)
                            + " — Created: " + subset[cols_map["date_created"]].astype(str)
                        )
                        options = subset["override_id"].tolist()
                        labels_map = dict(zip(options, subset["Label"].tolist()))

                        # Selected overrides (keep existing ones in this harvester group)
                        existing = set(ss.get("harv_overrides", set()))
                        default_vals = [oid for oid in options if oid in existing]

                        selected = st.multiselect(
                            "Select rows to **count as Harvester sits**:",
                            options=options,
                            default=default_vals,
                            format_func=lambda oid: labels_map.get(oid, oid),
                            key=f"ov_ms_{sel_harv}"
                        )

                        # Update override set (for current group only)
                        new_overrides = (existing - set(options)) | set(selected)
                        ss["harv_overrides"] = set(new_overrides)

                        # ---- Notes (required) for selected overrides ----
                        st.markdown("#### ✍️ Notes (required for each selected override)")
                        notes_map = dict(ss.get("harv_override_notes", {}))  # copy
                        for oid in selected:
                            # stable key per oid (shortened for Streamlit key length safety)
                            key = f"note_{hash(oid)}"
                            default_note = notes_map.get(oid, "")
                            note_text = st.text_input(
                                f"Reason for overriding: {labels_map.get(oid, oid)}",
                                value=default_note,
                                key=key
                            )
                            notes_map[oid] = note_text
                        # For non-selected in this group, keep existing notes as-is
                        ss["harv_override_notes"] = notes_map

                        # Quick drilldown with current valid overrides applied
                        st.markdown("#### 🔎 Drilldown for selected Harvester (after overrides)")
                        d1, d2 = st.columns(2)
                        hf_sit  = d1.checkbox("Only Sits (Harvester logic)", key="harv_sits_filter")
                        hf_sale = d2.checkbox("Only Sales", key="harv_sales_filter")

                        # Recompute effect with only overrides that have notes
                        valid_overrides = {oid for oid in ss["harv_overrides"] if ss["harv_override_notes"].get(oid, "").strip()}
                        df_effective2 = apply_harvester_overrides(df_tagged, valid_overrides)
                        mask2 = (df_effective2["Harvester"].astype(str) == sel_harv)
                        if hf_sit:  mask2 &= df_effective2["is_sit_harvester"]
                        if hf_sale: mask2 &= df_effective2["is_sale"]
                        drill2 = inject_override_cols(df_effective2.loc[mask2, show_cols + ["override_id"]])
                        st.dataframe(drill2.drop(columns=["override_id"]).reset_index(drop=True), use_container_width=True)

            # --- Harvester Pay (with overrides) ---
            with t4:
                st.dataframe(harv_pay_effective, use_container_width=True)

            # --- Print / Export ---
            with t5:
                st.markdown("### 🖨️ Print-friendly view")
                st.caption("Use your browser’s **File → Print** (or **Cmd/Ctrl+P**). This view shows summaries and a list of your override notes.")

                # Nicely formatted on-page print view
                st.subheader("Company Totals")
                st.dataframe(_format_percent_columns(company), use_container_width=True)
                st.subheader("Sales Rep Summary")
                st.dataframe(_format_percent_columns(rep_summary), use_container_width=True)
                st.subheader("Harvester Summary (with overrides)")
                st.dataframe(_format_percent_columns(harv_summary_effective), use_container_width=True)
                st.subheader("Harvester Pay (with overrides)")
                st.dataframe(harv_pay_effective, use_container_width=True)

                st.subheader("Override Notes")
                if not overrides_table.empty:
                    show_overrides = overrides_table.drop(columns=["override_id"])
                    st.dataframe(show_overrides, use_container_width=True)
                else:
                    st.write("No overrides applied.")

                # Download an HTML file that is print-ready
                html_bytes = build_print_html(
                    company_df=_format_percent_columns(company),
                    rep_df=_format_percent_columns(rep_summary),
                    harv_df=_format_percent_columns(harv_summary_effective),
                    pay_df=harv_pay_effective,
                    overrides_table=overrides_table.drop(columns=["override_id"]) if not overrides_table.empty else overrides_table
                ).encode("utf-8")
                st.download_button(
                    "⬇️ Download Print-Ready HTML",
                    data=html_bytes,
                    file_name="US_Shingle_Weekly_Report_Print.html",
                    mime="text/html",
                    help="Open this file in your browser and print or Save as PDF."
                )

                # CSV downloads for overridden Harvester tables (include notes)
                # Attach override indicators/notes to raw rows export
                if not overrides_table.empty:
                    # Also supply a raw rows CSV of overridden records (for audit)
                    audit_mask = df_tagged["override_id"].isin(valid_overrides)
                    audit = df_tagged.loc[audit_mask, [
                        cols_map["job_name"], cols_map["start_date"], cols_map["date_created"],
                        cols_map["status"], cols_map["sales_rep"], "Harvester",
                        cols_map["total_contract"], "override_id"
                    ]].copy()
                    audit["Override Note"] = audit["override_id"].map(lambda oid: notes_map.get(oid, ""))
                    audit = audit.rename(columns={
                        cols_map["job_name"]: "Job Name",
                        cols_map["start_date"]: "Start Date",
                        cols_map["date_created"]: "Date Created",
                        cols_map["status"]: "Status",
                        cols_map["sales_rep"]: "Sales Rep",
                        cols_map["total_contract"]: "Total Contract",
                    })
                    st.download_button(
                        "⬇️ Download Overridden Rows + Notes (CSV)",
                        data=audit.drop(columns=["override_id"]).to_csv(index=False).encode("utf-8"),
                        file_name="Harvester_Overrides_With_Notes.csv",
                        mime="text/csv"
                    )

    else:
        st.info("Upload a file to generate your report (top of this tab).")
