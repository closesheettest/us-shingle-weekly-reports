from pathlib import Path
import json
import io
import pandas as pd


# ---------- Config loading ----------
def load_config_from_path(config_path: Path):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg["_sit_statuses_lc"] = [s.lower() for s in cfg.get("sit_statuses", [])]
    cfg["_sale_statuses_lc"] = [s.lower() for s in cfg.get("sale_statuses", [])]
    cfg["_no_show_statuses_lc"] = [s.lower() for s in cfg.get("no_show_statuses", [])]
    cfg["_truthy_values"] = set(["true", "yes", "1"])
    return cfg


# ---------- File reader ----------
def read_input_file(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
    try:
        return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception:
        uploaded_file.seek(0)
    try:
        return pd.read_excel(uploaded_file, engine="xlrd")
    except Exception:
        uploaded_file.seek(0)
    try:
        return pd.read_excel(uploaded_file, engine="pyxlsb")
    except Exception:
        pass
    raise ValueError("Could not read file. Save as CSV or Excel (.xlsx).")


# ---------- Helpers ----------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def coerce_bool(series: pd.Series, truthy: set) -> pd.Series:
    def to_bool(x):
        if pd.isna(x):
            return False
        s = str(x).strip().lower()
        return s in truthy
    return series.apply(to_bool)


def ensure_columns(df: pd.DataFrame, cfg):
    def norm(s):
        return " ".join(str(s).replace("\u00A0", " ").strip().lower().split())
    wanted = cfg["columns"]
    norm_df = {norm(c): c for c in df.columns}
    resolved, missing = {}, []
    for key, wanted_name in wanted.items():
        n = norm(wanted_name)
        if n in norm_df:
            resolved[key] = norm_df[n]
        else:
            missing.append(wanted_name)
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    return resolved


def safe_div(n, d):
    return (n / d) if d else 0.0


def to_currency_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce"
    ).fillna(0.0)


# ---------- Status tagging ----------
def tag_statuses(df: pd.DataFrame, status_col: str, cfg) -> pd.DataFrame:
    status_lc = df[status_col].astype(str).str.strip().str.lower()
    df = df.copy()
    # Sales rules
    df["is_sit_sales"] = status_lc.isin(cfg["_sit_statuses_lc"])
    df["is_sale"] = status_lc.isin(cfg["_sale_statuses_lc"])
    df["is_no_show"] = status_lc.isin(cfg["_no_show_statuses_lc"])
    # Harvester rules (same + “new roof”)
    df["is_sit_harvester"] = df["is_sit_sales"] | status_lc.eq("new roof")
    return df


# ---------- Reports ----------
def compute_sales_rep_summary(df: pd.DataFrame, cols, truthy: set) -> pd.DataFrame:
    amt_col = cols["total_contract"]
    ins_col = cols["insulation"]
    rb_col  = cols["radiant_barrier"]
    rep_col = cols["sales_rep"]

    # Coerce add-ons to booleans (True/Yes/1 -> True)
    df_ins = coerce_bool(df[ins_col], truthy)
    df_rb  = coerce_bool(df[rb_col], truthy)

    # GROUP BY sales rep
    grp = df.groupby(rep_col, dropna=False)

    # $ amount only from rows that are sales, cleaned as currency
    def sales_amount_sum(g: pd.DataFrame) -> float:
        sold_vals = to_currency_numeric(g.loc[g["is_sale"], amt_col])
        return float(sold_vals.sum())

    # Base counts
    summary = pd.DataFrame({
        "Total Appointments": grp.size(),
        "Total Sits": grp["is_sit_sales"].sum(),   # <-- Sales sit logic
        "Total Sales": grp["is_sale"].sum(),
        "$ Sales Amount": grp.apply(sales_amount_sum),
    })

    # Rates
    summary["Sit %"] = (summary["Total Sits"] / summary["Total Appointments"]).fillna(0)
    summary["Sales %"] = (summary["Total Sales"] / summary["Total Sits"]).fillna(0)
    no_shows = grp["is_no_show"].sum()
    summary["No Show %"] = (no_shows / summary["Total Appointments"]).fillna(0)

    # Average ticket
    summary["Avg $ / Sale"] = summary.apply(
        lambda r: safe_div(r["$ Sales Amount"], r["Total Sales"]), axis=1
    )

    # Add-on % should be "of Sales" (denominator = number of sales)
    def pct_addon_of_sales(g: pd.DataFrame, addon_bool: pd.Series) -> float:
        sold = g[g["is_sale"]]
        if sold.empty:
            return 0.0
        addon_on_sold = addon_bool.loc[sold.index]
        return float(addon_on_sold.sum()) / len(sold)

    summary["Insulation % (of Sales)"]      = grp.apply(lambda g: pct_addon_of_sales(g, df_ins))
    summary["Radiant Barrier % (of Sales)"] = grp.apply(lambda g: pct_addon_of_sales(g, df_rb))

    return summary.reset_index(names=[rep_col])


    def sales_amount_sum(g: pd.DataFrame) -> float:
        return float(to_currency_numeric(g.loc[g["is_sale"], amt_col]).sum())

    summary = pd.DataFrame({
        "Total Appointments": grp.size(),
        "Total Sits": grp["is_sit_sales"].sum(),
        "Total Sales": grp["is_sale"].sum(),
        "$ Sales Amount": grp.apply(sales_amount_sum),
    })
    summary["Sit %"] = (summary["Total Sits"] / summary["Total Appointments"]).fillna(0)
    summary["Sales %"] = (summary["Total Sales"] / summary["Total Sits"]).fillna(0)
    summary["No Show %"] = (grp["is_no_show"].sum() / summary["Total Appointments"]).fillna(0)
    summary["Avg $ / Sale"] = summary.apply(lambda r: safe_div(r["$ Sales Amount"], r["Total Sales"]), axis=1)
    summary["Insulation % (of Sales)"] = grp.apply(lambda g: (coerce_bool(g[ins_col], truthy) & g["is_sale"]).mean())
    summary["Radiant Barrier % (of Sales)"] = grp.apply(lambda g: (coerce_bool(g[rb_col], truthy) & g["is_sale"]).mean())
    return summary.reset_index(names=[rep_col])


def compute_company_totals(raw_df: pd.DataFrame, cols) -> pd.DataFrame:
    amt_col = cols["total_contract"]
    totals = {
        "Total Appointments": int(raw_df.shape[0]),
        "Total Sits": int(raw_df["is_sit_sales"].sum()),
        "Total Sales": int(raw_df["is_sale"].sum())
    }
    totals["$ Sales Amount"] = float(to_currency_numeric(raw_df.loc[raw_df["is_sale"], amt_col]).sum())
    totals["Sit %"] = safe_div(totals["Total Sits"], totals["Total Appointments"])
    totals["Sales %"] = safe_div(totals["Total Sales"], totals["Total Sits"])
    totals["No Show %"] = safe_div(int(raw_df["is_no_show"].sum()), totals["Total Appointments"])
    totals["Avg $ / Sale"] = safe_div(totals["$ Sales Amount"], totals["Total Sales"])
    return pd.DataFrame([totals])


def compute_harvester_report(df: pd.DataFrame, cols) -> pd.DataFrame:
    harvester = df[cols["appointment_set_by"]].fillna("").astype(str).str.strip().replace("", "Company")
    df2 = df.copy(); df2["Harvester"] = harvester
    grp = df2.groupby("Harvester", dropna=False)
    harv = pd.DataFrame({
        "Appointments Set": grp.size(),
        "Sits": grp["is_sit_harvester"].sum(),
    })
    harv["Sit %"] = (harv["Sits"] / harv["Appointments Set"]).fillna(0)
    return harv.reset_index()


def compute_harvester_pay(harvester_report: pd.DataFrame) -> pd.DataFrame:
    def pay_for_sits(n):
        n = int(n or 0)
        if n < 8: return 100 * n, 0, 100
        elif n == 8: return 100 * 8 + 500, 500, 100
        elif 9 <= n <= 14: return 125 * n + 500, 500, 125
        else: return 150 * n + 500, 500, 150
    rows = []
    for _, r in harvester_report.iterrows():
        pay, bonus, rate = pay_for_sits(r["Sits"])
        rows.append({"Harvester": r["Harvester"], "Sits": int(r["Sits"]), "Base Rate Applied": rate, "Bonus (if any)": bonus, "Total Pay": pay})
    return pd.DataFrame(rows)


# ---------- Excel writer ----------
def build_workbook(df: pd.DataFrame, cfg: dict) -> bytes:
    df = normalize_columns(df)
    cols = ensure_columns(df, cfg)
    df = tag_statuses(df, cols["status"], cfg)

    rep_summary = compute_sales_rep_summary(df, cols, cfg["_truthy_values"])
    company = compute_company_totals(df, cols)
    harvester = compute_harvester_report(df, cols)
    harvester_pay = compute_harvester_pay(harvester)

    output_stream = io.BytesIO()
    with pd.ExcelWriter(output_stream, engine="xlsxwriter") as writer:
        rep_summary.to_excel(writer, sheet_name="Sales Rep Summary", index=False)
        company.to_excel(writer, sheet_name="Company Totals", index=False)
        harvester.to_excel(writer, sheet_name="Harvester Summary", index=False)
        harvester_pay.to_excel(writer, sheet_name="Harvester Pay", index=False)

        workbook = writer.book
        percent_fmt = workbook.add_format({"num_format": "0%"})
        money_fmt = workbook.add_format({"num_format": "$#,##0"})
        money2_fmt = workbook.add_format({"num_format": "$#,##0.00"})

        for col, name in enumerate(rep_summary.columns):
            if name.endswith("%"): writer.sheets["Sales Rep Summary"].set_column(col, col, 12, percent_fmt)
        for col, name in enumerate(company.columns):
            if name.endswith("%"): writer.sheets["Company Totals"].set_column(col, col, 12, percent_fmt)
        if "Sit %" in harvester.columns:
            idx = list(harvester.columns).index("Sit %")
            writer.sheets["Harvester Summary"].set_column(idx, idx, 12, percent_fmt)

    output_stream.seek(0)
    return output_stream.getvalue()
