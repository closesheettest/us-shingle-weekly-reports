import io
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from report_utils import (
    load_default_config,
    coerce_money,
    build_sales_by_rep_table,
    build_harvester_table,
    build_harvester_pay_table,
)

st.set_page_config(page_title="US Shingle Weekly Reports", layout="wide")

st.title("US Shingle Weekly Reports")
st.caption("Single upload → map columns & statuses → Sales Rep & Harvester metrics. NET metrics exclude statuses you mark as Credit Denial.")

# ---------- Upload ----------
upl = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xlsm", "xls"])

if "df" not in st.session_state:
    st.session_state.df = None
if "cfg" not in st.session_state:
    st.session_state.cfg = load_default_config()

if upl:
    try:
        if upl.name.lower().endswith(".csv"):
            df = pd.read_csv(upl)
        else:
            df = pd.read_excel(upl)
        st.session_state.df = df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

df = st.session_state.df
cfg = st.session_state.cfg

with st.expander("Settings", expanded=False):
    st.markdown("#### Column mapping")
    if df is None:
        st.info("Upload a file to configure mapping.")
    else:
        cols = list(df.columns)
        colmap = cfg["column_map"]
        # map each required semantic field to one of the file columns
        def map_select(label, key_name, help_txt=""):
            colmap[key_name] = st.selectbox(
                label, options=["— Not in file —"] + cols,
                index=(cols.index(colmap.get(key_name)) + 1) if colmap.get(key_name) in cols else 0,
                help=help_txt,
                key=f"map_{key_name}",
            )
        map_select("Job Name", "job_name")
        map_select("Date Created", "date_created")
        map_select("Start Date", "start_date")
        map_select("Status", "status")
        map_select("Sales Rep", "sales_rep")
        map_select("Total Contract ($)", "total_contract", "Sales $ column")
        map_select("Appointment Set By (Harvester)", "harvester")
        map_select("Insulation Cost ($)", "insulation_cost")
        map_select("Radiant Barrier Cost ($)", "radiant_barrier_cost")

        # status tagging
        st.markdown("#### Status tagging")
        status_col = colmap.get("status")
        if status_col and status_col in df.columns:
            all_statuses = sorted([str(x) for x in df[status_col].dropna().unique()])
        else:
            all_statuses = []

        def tag_multiselect(label, cfg_key, help_txt=""):
            current = [s for s in cfg[cfg_key] if s in all_statuses]
            cfg[cfg_key] = st.multiselect(
                label,
                options=all_statuses,
                default=current,
                help=help_txt,
                key=f"tag_{cfg_key}",
            )

        tag_multiselect("**Sale (Sales)** statuses", "sale_statuses", "These count as SALES for Sales Reps.")
        tag_multiselect("**Sit (Sales)** statuses", "sit_sales_statuses", "These count as SITS for Sales Reps.")
        tag_multiselect("**Sit (Harvester)** statuses", "sit_harvest_statuses", "These count as SITS for Harvesters.")
        tag_multiselect("**Credit Denial** statuses (excluded from NET)", "credit_denial_statuses",
                        "Rows with these statuses are removed from NET metrics entirely.")

        # Save settings button
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Save settings (in session)"):
                st.session_state.cfg = cfg
                st.success("Saved in session.")
        with c2:
            # download current settings
            st.download_button(
                "Download settings.json",
                data=json.dumps(cfg, indent=2),
                file_name="settings.json",
                mime="application/json",
            )

# no file -> stop
if df is None:
    st.stop()

# ---------- Normalize / prepare ----------
colmap = cfg["column_map"].copy()

# keep only mapped columns that exist
internal_cols = {
    "job_name": colmap.get("job_name"),
    "date_created": colmap.get("date_created"),
    "start_date": colmap.get("start_date"),
    "status": colmap.get("status"),
    "sales_rep": colmap.get("sales_rep"),
    "total_contract": colmap.get("total_contract"),
    "harvester": colmap.get("harvester"),
    "insulation_cost": colmap.get("insulation_cost"),
    "radiant_barrier_cost": colmap.get("radiant_barrier_cost"),
}
present = {k: v for k, v in internal_cols.items() if v and v in df.columns}

work = df.rename(columns={v: k for k, v in present.items()}).copy()

# Make sure numeric money fields are numeric
for money_col in ["total_contract", "insulation_cost", "radiant_barrier_cost"]:
    if money_col in work.columns:
        work[money_col] = coerce_money(work[money_col])

# dates (optional)
for dcol in ["date_created", "start_date"]:
    if dcol in work.columns:
        work[dcol] = pd.to_datetime(work[dcol], errors="coerce")

# status strings
if "status" in work.columns:
    work["status"] = work["status"].astype(str)
else:
    st.error("You must map the Status column in Settings.")
    st.stop()

# ---------- Build reports ----------
sale_statuses = set(cfg["sale_statuses"])
sit_sales_statuses = set(cfg["sit_sales_statuses"])
sit_harvest_statuses = set(cfg["sit_harvest_statuses"])
credit_denial_statuses = set(cfg["credit_denial_statuses"])

sales_by_rep = build_sales_by_rep_table(
    work,
    sale_statuses=sale_statuses,
    sit_sales_statuses=sit_sales_statuses,
    credit_denial_statuses=credit_denial_statuses,
)

harvester_tbl = build_harvester_table(
    work,
    sit_harvest_statuses=sit_harvest_statuses,
)

pay_tbl = build_harvester_pay_table(harvester_tbl)

# ---------- KPIs ----------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Appointments", int(sales_by_rep["Appointments"].sum()))
k2.metric("Sits", int(sales_by_rep["Sits"].sum()))
k3.metric("Sales", int(sales_by_rep["Sales"].sum()))
k4.metric("Net Sits (adj.)", int(sales_by_rep["Net Sits"].sum()))
k5.metric("Net Sales (adj.)", int(sales_by_rep["Net Sales"].sum()))

st.markdown("### By Sales Rep")
st.dataframe(sales_by_rep, use_container_width=True)

st.markdown("### By Harvester")
st.dataframe(harvester_tbl, use_container_width=True)

st.markdown("### Harvester Pay (weekly)")
st.dataframe(pay_tbl, use_container_width=True)

# ---------- Downloads ----------
out = io.BytesIO()
with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
    sales_by_rep.to_excel(writer, sheet_name="Sales by Rep", index=False)
    harvester_tbl.to_excel(writer, sheet_name="Harvesters", index=False)
    pay_tbl.to_excel(writer, sheet_name="Harvester Pay", index=False)

st.download_button("Download Excel", data=out.getvalue(), file_name=f"weekly_reports_{datetime.now():%Y%m%d}.xlsx")
