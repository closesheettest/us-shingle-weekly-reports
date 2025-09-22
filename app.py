# app.py — 'Sales miscount fix' look restored + persistent status mapping
from __future__ import annotations
import io, re
import streamlit as st
import pandas as pd
from typing import Dict, Any

from status_persist import load_status_rules, save_status_rules, STATUS_RULES_PATH

st.set_page_config(page_title="US Shingle Weekly Reports", layout="wide")

# ---------- Canonical mapping record ----------
CANON_EMPTY = {"sales": {"sale": False, "sit": False}, "harvester": {"sit": False}}
def canon(rec: Any) -> Dict[str, Any]:
    if not isinstance(rec, dict):
        return dict(CANON_EMPTY)
    s = rec.get("sales", {}) if isinstance(rec, dict) else {}
    h = rec.get("harvester", {}) if isinstance(rec, dict) else {}
    return {
        "sales": {"sale": bool(s.get("sale", False)), "sit": bool(s.get("sit", False))},
        "harvester": {"sit": bool(h.get("sit", False))}
    }

# Persisted rules
rules: Dict[str, Any] = {k: canon(v) for k, v in load_status_rules().items()}

# ---------- Sidebar + shared upload ----------
st.sidebar.title("US Shingle")
page = st.sidebar.radio(
    "Go to",
    ["Mapping Editor", "Harvester Results", "Sales Results by Rep", "Neal's Pay"],
    index=0
)

def get_df() -> pd.DataFrame | None:
    if "uploaded_df" not in st.session_state:
        st.session_state.uploaded_df = None
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="csv_global")
    if up is not None:
        try:
            st.session_state.uploaded_df = pd.read_csv(io.BytesIO(up.read()))
        except Exception as e:
            st.sidebar.error(f"CSV error: {e}")
            st.session_state.uploaded_df = None
    return st.session_state.uploaded_df

df = get_df()
if df is not None:
    st.sidebar.caption(f"Rows: {len(df):,} | Cols: {len(df.columns)}")

# Column pickers (no hardcoding)
def pick_columns(df: pd.DataFrame, purpose_key: str):
    cols = list(df.columns)
    def _idx(candidates, default=0):
        lower = [c.lower() for c in cols]
        for name in candidates:
            if name in lower:
                return lower.index(name)
        return default
    st.markdown("**Select columns**")
    status_col = st.selectbox("Status column", options=cols,
                              index=_idx({"status","job status","appointment status"}), key=f"{purpose_key}_status")
    sales_col = st.selectbox("Sales rep column", options=cols,
                             index=_idx({"sales rep","rep","salesperson","closer"}), key=f"{purpose_key}_sales")
    harv_col  = st.selectbox("Harvester rep column", options=cols,
                             index=_idx({"harvester","setter","canvasser"}), key=f"{purpose_key}_harv")
    return status_col, sales_col, harv_col

def apply_mapping(df: pd.DataFrame, status_col: str) -> pd.DataFrame:
    out = df.copy()
    out["sales_sale"]    = out[status_col].astype(str).map(lambda v: canon(rules.get(v, {}))["sales"]["sale"])
    out["sales_sit"]     = out[status_col].astype(str).map(lambda v: canon(rules.get(v, {}))["sales"]["sit"])
    out["harvester_sit"] = out[status_col].astype(str).map(lambda v: canon(rules.get(v, {}))["harvester"]["sit"])
    return out

# =======================================
# Mapping Editor
# =======================================
if page == "Mapping Editor":
    st.title("Mapping Editor")
    st.caption(f"Mapping file: {STATUS_RULES_PATH} • Loaded statuses: {len(rules)}")

    if df is not None:
        cols = list(df.columns)
        status_idx = 0
        for c in cols:
            if "status" in c.lower():
                status_idx = cols.index(c); break
        status_col = st.selectbox("Add unseen from this Status column", options=cols, index=status_idx)
        if st.button("Scan & add NEW statuses"):
            uniques = sorted(pd.Series(df[status_col].astype(str)).fillna("").unique())
            to_add = [v for v in uniques if v and v not in rules]
            for v in to_add:
                rules[v] = dict(CANON_EMPTY)
            save_status_rules(rules)
            st.success(f"Added {len(to_add)} new statuses.")

    def is_unmapped(rec: Dict[str, Any]) -> bool:
        c = canon(rec); return not (c["sales"]["sale"] or c["sales"]["sit"] or c["harvester"]["sit"])

    names_sorted = sorted(rules.keys(), key=lambda k: (not is_unmapped(rules[k]), k.lower()))
    changed = False
    for raw in names_sorted:
        rec = canon(rules[raw])
        c1, c2, c3, c4 = st.columns([2,2,2,2])
        with c1: st.markdown(f"**{raw}**{'  🆕' if is_unmapped(rec) else ''}")
        with c2: sale_val = st.checkbox("Sale (Sales)", value=rec["sales"]["sale"], key=f"{hash((raw,'sale'))}")
        with c3: sit_val  = st.checkbox("Sit (Sales)",  value=rec["sales"]["sit"],  key=f"{hash((raw,'sit'))}")
        with c4: harv_val = st.checkbox("Sit (Harvester)", value=rec["harvester"]["sit"], key=f"{hash((raw,'harv'))}")
        new_rec = {"sales": {"sale": bool(sale_val), "sit": bool(sit_val)}, "harvester": {"sit": bool(harv_val)}}
        if new_rec != rec:
            rules[raw] = new_rec; changed = True

    if st.button("Save Mapping"):
        if save_status_rules(rules): st.success("Saved ✅")
        else: st.info("No changes to save")


# =======================================
# Harvester Results
# =======================================
elif page == "Harvester Results":
    st.title("Harvester Results")
    if df is None:
        st.info("Upload a CSV in the sidebar.")
    else:
        status_col, _sales_col, harv_col = pick_columns(df, "harv")
        mapped = apply_mapping(df, status_col)

        st.subheader("KPIs")
        total_rows = len(mapped)
        harv_sits = int(mapped["harvester_sit"].sum())
        sales_all = int(mapped["sales_sale"].sum())
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Harvester Sits", harv_sits)
        with c2: st.metric("Sales (all)", sales_all)
        with c3: st.metric("Rows", total_rows)

        st.subheader("By Harvester")
        grp = (
            mapped.groupby(harv_col, dropna=False)
                  .agg(Harvester_Sits=("harvester_sit","sum"),
                       Sales=("sales_sale","sum"),
                       Sales_Sits=("sales_sit","sum"),
                       Count=(harv_col,"count"))
                  .reset_index()
                  .sort_values("Harvester_Sits", ascending=False)
        )
        st.dataframe(grp, use_container_width=True)
        st.download_button("Download harvester results CSV",
                           grp.to_csv(index=False).encode("utf-8"),
                           "harvester_results.csv", "text/csv")


# =======================================
# Sales Results by Rep (Gross & Net %)
# =======================================
elif page == "Sales Results by Rep":
    st.title("Sales Results by Rep")
    if df is None:
        st.info("Upload a CSV in the sidebar.")
    else:
        status_col, sales_col, _harv_col = pick_columns(df, "sales")
        mapped = apply_mapping(df, status_col)

        # Gross = Sales / Sales Sits
        sales_cnt = int(mapped["sales_sale"].sum())
        sits_sales_cnt = int(mapped["sales_sit"].sum())
        gross_pct = (sales_cnt / sits_sales_cnt * 100.0) if sits_sales_cnt else 0.0

        # Net = Sales / Net Sits (exclude obvious 'no sit' statuses; adjustable)
        with st.expander("Net Closing % settings", expanded=False):
            kw_excl_on = st.checkbox("Exclude 'no sit' keywords (no show / cancel / resched / credit / denial)", value=True)
            kw_pattern = r"(no\s*show|no-show|cancel|resched|credit|denial)"
            manual_excl = st.multiselect("Additionally exclude these raw Status values:",
                                         sorted(mapped[status_col].astype(str).fillna("").unique()), default=[])

        kw_mask = (
            mapped[status_col].astype(str).str.lower().str.contains(kw_pattern, flags=re.IGNORECASE, regex=True)
            if kw_excl_on else pd.Series(False, index=mapped.index)
        )
        manual_mask = mapped[status_col].astype(str).isin(set(manual_excl))
        net_sits_mask = mapped["sales_sit"] & ~(kw_mask | manual_mask)
        net_sits_cnt = int(net_sits_mask.sum())
        net_pct = (sales_cnt / net_sits_cnt * 100.0) if net_sits_cnt else 0.0

        st.subheader("KPIs")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("Sales Sits", sits_sales_cnt)
        with c2: st.metric("Sales (closed)", sales_cnt)
        with c3: st.metric("Gross Closing %", f"{gross_pct:.1f}%")
        with c4: st.metric("Net Sits (adj.)", net_sits_cnt)
        with c5: st.metric("Net Closing %", f"{net_pct:.1f}%")

        st.subheader("By Sales Rep")
        by_rep = (
            mapped.assign(net_sit=net_sits_mask)
                  .groupby(sales_col, dropna=False)
                  .agg(Sales_Sits=("sales_sit","sum"),
                       Net_Sits=("net_sit","sum"),
                       Sales=("sales_sale","sum"),
                       Count=(sales_col,"count"))
                  .reset_index()
        )
        by_rep["Gross Closing %"] = by_rep.apply(lambda r: (r["Sales"]/r["Sales_Sits"]*100.0) if r["Sales_Sits"] else 0.0, axis=1)
        by_rep["Net Closing %"] = by_rep.apply(lambda r: (r["Sales"]/r["Net_Sits"]*100.0) if r["Net_Sits"] else 0.0, axis=1)
        by_rep = by_rep.sort_values(["Sales","Sales_Sits"], ascending=[False, False])
        st.dataframe(by_rep, use_container_width=True)

        st.download_button("Download sales-by-rep CSV",
                           by_rep.to_csv(index=False).encode("utf-8"),
                           "sales_by_rep.csv", "text/csv")


# =======================================
# Neal's Pay
# =======================================
elif page == "Neal's Pay":
    st.title("Neal’s Pay")
    if df is None:
        st.info("Upload a CSV in the sidebar.")
    else:
        status_col, sales_col, harv_col = pick_columns(df, "pay")
        mapped = apply_mapping(df, status_col)

        st.subheader("Pay Settings")
        colA, colB, colC = st.columns(3)
        with colA:
            pay_per_sale = st.number_input("Pay per SALE ($)", min_value=0.0, value=100.0, step=10.0)
        with colB:
            bonus_per_sales_sit = st.number_input("Bonus per Sales SIT ($)", min_value=0.0, value=0.0, step=5.0)
        with colC:
            harv_bonus_per_sit = st.number_input("Harvester bonus per SIT ($)", min_value=0.0, value=25.0, step=5.0)

        sales_by_rep = (
            mapped.groupby(sales_col, dropna=False)
                  .agg(Sales=("sales_sale","sum"),
                       Sales_Sits=("sales_sit","sum"))
                  .reset_index()
        )
        sales_by_rep["Rep Payout ($)"] = sales_by_rep["Sales"] * pay_per_sale + sales_by_rep["Sales_Sits"] * bonus_per_sales_sit

        harv_by_rep = (
            mapped.groupby(harv_col, dropna=False)
                  .agg(Harvester_Sits=("harvester_sit","sum"))
                  .reset_index()
        )
        harv_by_rep["Harvester Payout ($)"] = harv_by_rep["Harvester_Sits"] * harv_bonus_per_sit

        st.subheader("Sales Rep Payouts")
        st.dataframe(sales_by_rep.sort_values("Rep Payout ($)", ascending=False), use_container_width=True)
        st.download_button("Download sales rep payouts CSV",
                           sales_by_rep.to_csv(index=False).encode("utf-8"),
                           "neals_pay_sales.csv", "text/csv")

        st.subheader("Harvester Payouts")
        st.dataframe(harv_by_rep.sort_values("Harvester Payout ($)", ascending=False), use_container_width=True)
        st.download_button("Download harvester payouts CSV",
                           harv_by_rep.to_csv(index=False).encode("utf-8"),
                           "neals_pay_harvesters.csv", "text/csv")
