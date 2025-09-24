import numpy as np
import pandas as pd


def load_default_config():
    # Minimal defaults – no hardcoded statuses to avoid "default not in options" errors
    return {
        "column_map": {
            "job_name": "",
            "date_created": "",
            "start_date": "",
            "status": "",
            "sales_rep": "",
            "total_contract": "",
            "harvester": "",
            "insulation_cost": "",
            "radiant_barrier_cost": "",
        },
        "sale_statuses": [],
        "sit_sales_statuses": [],
        "sit_harvest_statuses": [],
        "credit_denial_statuses": [],
    }


def coerce_money(s):
    """Safely coerce strings like '$1,234.50' to numeric."""
    return (
        pd.to_numeric(
            s.astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .str.strip(),
            errors="coerce",
        )
        .fillna(0.0)
    )


def _format_pct(series):
    """Return whole-number percentages as strings without decimals."""
    # Avoid division by zero / NaN
    out = (series * 100).round(0).astype("Int64")
    return out.astype(str) + "%"


def _fmt_money(series):
    return series.round(2).map(lambda x: f"${x:,.2f}")


def build_sales_by_rep_table(
    df,
    sale_statuses: set,
    sit_sales_statuses: set,
    credit_denial_statuses: set,
):
    """Build the per-rep table in the exact column order requested."""
    work = df.copy()

    # helper flags
    work["is_appt"] = True  # each row is an appointment
    work["is_sit_sales"] = work["status"].isin(sit_sales_statuses)
    work["is_sale"] = work["status"].isin(sale_statuses)
    work["is_credit_denial"] = work["status"].isin(credit_denial_statuses)

    # dollar columns
    for col in ["total_contract", "insulation_cost", "radiant_barrier_cost"]:
        if col not in work.columns:
            work[col] = 0.0

    # NET filter removes credit denial rows entirely
    net = work.loc[~work["is_credit_denial"]].copy()

    # counts by rep
    grp_all = work.groupby("sales_rep", dropna=False)
    grp_net = net.groupby("sales_rep", dropna=False)

    # appointments
    appts = grp_all["is_appt"].sum(min_count=1).fillna(0).astype(int)

    # sits / net sits
    sits = grp_all["is_sit_sales"].sum(min_count=1).fillna(0).astype(int)
    net_sits = grp_net["is_sit_sales"].sum(min_count=1).fillna(0).astype(int)

    # sales / net sales (count)
    sales = grp_all["is_sale"].sum(min_count=1).fillna(0).astype(int)
    net_sales = grp_net["is_sale"].sum(min_count=1).fillna(0).astype(int)

    # close rates
    sit_pct = (sits / appts.replace(0, np.nan)).fillna(0.0)
    net_sit_pct = (net_sits / grp_net["is_appt"].sum(min_count=1).replace(0, np.nan)).fillna(0.0)
    sales_pct = (sales / sits.replace(0, np.nan)).fillna(0.0)
    net_sales_pct = (net_sales / net_sits.replace(0, np.nan)).fillna(0.0)

    # dollars
    insul = grp_all["insulation_cost"].sum(min_count=1).fillna(0.0)
    rb = grp_all["radiant_barrier_cost"].sum(min_count=1).fillna(0.0)

    # IRBAD % = percent of *sales count* with at least one of the two
    work["has_irbad"] = (work["insulation_cost"] > 0) | (work["radiant_barrier_cost"] > 0)
    sales_only = work.loc[work["is_sale"]].copy()
    grp_sales_only = sales_only.groupby("sales_rep", dropna=False)
    irbad_count = grp_sales_only["has_irbad"].sum(min_count=1).reindex(appts.index).fillna(0).astype(int)
    sales_count = sales.reindex(appts.index).fillna(0).astype(int)
    irbad_pct = (irbad_count / sales_count.replace(0, np.nan)).fillna(0.0)

    # assemble
    out = pd.DataFrame({
        "Sales Rep": appts.index.astype(str),
        "Appointments": appts.values,
        "Sits": sits.values,
        "Sit %": _format_pct(sit_pct),
        "Net Sits": net_sits.values,
        "Net Sit %": _format_pct(net_sit_pct),
        "Sales": sales.values,
        "Sales %": _format_pct(sales_pct),
        "Net Sales": net_sales.values,
        "Net Sales %": _format_pct(net_sales_pct),
        "Insulation $": _fmt_money(insul),
        "Radiant Barrier $": _fmt_money(rb),
        "IRBAD %": _format_pct(irbad_pct),
    }).sort_values("Sales Rep", kind="stable").reset_index(drop=True)

    return out


def build_harvester_table(df, sit_harvest_statuses: set):
    work = df.copy()
    work["is_appt"] = True
    work["is_sit_harv"] = work["status"].isin(sit_harvest_statuses)

    grp = work.groupby("harvester", dropna=False)
    appts = grp["is_appt"].sum(min_count=1).fillna(0).astype(int)
    sits = grp["is_sit_harv"].sum(min_count=1).fillna(0).astype(int)
    sit_pct = (sits / appts.replace(0, np.nan)).fillna(0.0)

    out = pd.DataFrame({
        "Harvester": appts.index.astype(str),
        "Appointments": appts.values,
        "Sits": sits.values,
        "Sit %": _format_pct(sit_pct),
    }).sort_values("Harvester", kind="stable").reset_index(drop=True)
    return out


def _harvester_pay_formula(sits: int) -> float:
    """
    0–8 sits: $100 per sit, $500 bonus at 8.
    9–14 sits: $125 per sit from the beginning (retroactive) + $500 bonus.
    15+ sits: $150 per sit from the beginning (retroactive) + $500 bonus.
    """
    if sits <= 0:
        return 0.0
    if sits < 8:
        return sits * 100.0
    if 8 <= sits < 9:
        return sits * 100.0 + 500.0
    if 9 <= sits < 15:
        return sits * 125.0 + 500.0
    # 15 or more
    return sits * 150.0 + 500.0


def build_harvester_pay_table(harvester_tbl: pd.DataFrame) -> pd.DataFrame:
    t = harvester_tbl.copy()
    t["Pay ($)"] = t["Sits"].apply(_harvester_pay_formula)
    return t[["Harvester", "Sits", "Pay ($)"]]
