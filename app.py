def compute_source_report(flag_df: pd.DataFrame, source_col: str = "Source", sit_mode: str = "sales") -> pd.DataFrame:
    df = flag_df.copy()
    if source_col not in df.columns:
        df[source_col] = ""
    sit_flag = "is_sit_sales" if sit_mode == "sales" else "is_sit_harv"

    def row(g: pd.DataFrame):
        appts = len(g)
        sits  = int(g[sit_flag].sum())
        sales = int(g["is_sale"].sum())  # may include $>0 inference

        sale_mask = g.get("is_sale_status", pd.Series(False, index=g.index)).astype(bool)
        sales_amt = float(g.loc[sale_mask, "Total Contract"].sum()) if "Total Contract" in g.columns else 0.0
        strict_sales = int(sale_mask.sum())

        sales_with_insul = int(g["has_insul"].sum())
        sales_with_rb    = int(g["has_rb"].sum())
        sales_with_addon = int(g["has_any_addon"].sum())

        insul_cost_sum = float(g.loc[g["has_insul"], "Insulation Cost"].sum()) if "Insulation Cost" in g.columns else 0.0
        rb_cost_sum    = float(g.loc[g["has_rb"], "Radiant Barrier Cost"].sum()) if "Radiant Barrier Cost" in g.columns else 0.0

        return pd.DataFrame([{
            "Appointments": appts,
            "Sits": sits,
            "Sit %": (sits / appts) if appts else 0.0,
            "Sales": sales,
            "Close %": (sales / sits) if sits else 0.0,
            "Sales $": sales_amt,
            "Avg Sale $": (sales_amt / strict_sales) if strict_sales else 0.0,

            "Sales with Insulation #": sales_with_insul,
            "Insul % of Sales": (sales_with_insul / strict_sales) if strict_sales else 0.0,
            "Insul $": insul_cost_sum,

            "Sales with RB #": sales_with_rb,
            "RB % of Sales": (sales_with_rb / strict_sales) if strict_sales else 0.0,
            "RB $": rb_cost_sum,

            "Sales with Add-on #": sales_with_addon,
            "Add-on % of Sales": (sales_with_addon / strict_sales) if strict_sales else 0.0,
        }])

    rep = df.groupby(source_col, dropna=False).apply(row).reset_index(level=1, drop=True).reset_index().rename(columns={source_col: "Source"})
    return rep
