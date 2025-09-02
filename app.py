def build_flags(df: pd.DataFrame, *, rules: dict,
                infer_sale_from_amount: bool,
                count_sold_as_sit_harv: bool,
                count_sold_as_sit_sales: bool) -> pd.DataFrame:
    out = df.copy()
    status_clean = out["Status"].astype(str).str.strip()

    sit_harv_set = set(rules.get("sit_harvester", []))
    sit_sales_set = set(rules.get("sit_sales", []))
    sale_set      = set(rules.get("sale", []))
    noshow_set    = set(rules.get("no_show", []))

    # Strict sale flag: ONLY by status
    is_sale_status = status_clean.isin(sale_set)
    out["is_sale_status"] = is_sale_status

    # Backward-compatible sale flag (may include $>0 inference if enabled)
    is_sale_from_rule = is_sale_status
    is_sale_from_amt  = False
    if "Total Contract" in out.columns:
        is_sale_from_amt = pd.to_numeric(out["Total Contract"], errors="coerce").fillna(0) > 0
    out["is_sale"] = is_sale_from_rule | (infer_sale_from_amount & is_sale_from_amt)

    # Sit flags (separate)
    out["is_sit_harv"]  = status_clean.isin(sit_harv_set)  | (count_sold_as_sit_harv  & out["is_sale"])
    out["is_sit_sales"] = status_clean.isin(sit_sales_set) | (count_sold_as_sit_sales & out["is_sale"])

    # No Show
    out["is_noshow"] = status_clean.isin(noshow_set)

    # OVERRIDES → Harvester only
    if "Override To Sit" in out.columns:
        ov = out["Override To Sit"].fillna(False).astype(bool)
        out.loc[ov, "is_sit_harv"] = True

    # Insulation / RB attributes (count ONLY when status is a Sale)
    insul_cost = out.get("Insulation Cost", pd.Series(0, index=out.index))
    insul_sqft = out.get("Insulation Sqft", pd.Series(0, index=out.index))
    rb_cost    = out.get("Radiant Barrier Cost", pd.Series(0, index=out.index))
    rb_sqft    = out.get("Radiant Barrier Sqft", pd.Series(0, index=out.index))

    has_insul_any = (pd.to_numeric(insul_cost, errors="coerce").fillna(0) > 0) | \
                    (pd.to_numeric(insul_sqft, errors="coerce").fillna(0) > 0)
    has_rb_any    = (pd.to_numeric(rb_cost,  errors="coerce").fillna(0) > 0) | \
                    (pd.to_numeric(rb_sqft,  errors="coerce").fillna(0) > 0)

    # IMPORTANT: gate by is_sale_status (strict, by status)
    out["has_insul"]      = is_sale_status & has_insul_any
    out["has_rb"]         = is_sale_status & has_rb_any
    out["has_any_addon"]  = is_sale_status & (has_insul_any | has_rb_any)
    return out
