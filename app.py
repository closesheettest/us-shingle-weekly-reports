# app.py — US Shingle Weekly Reports
# - Status Rules editor lists ONLY statuses in current file; auto-save merges into status_rules.json
# - Overrides: "Override to Sit" affects HARVESTER ONLY (does NOT affect Sales-side sits)
# - Insulation/Radiant Barrier are attributes of sales
# - Adds Sources report + exports
# - Printable Harvester report with overrides & notes

import json, re
from io import BytesIO
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="US Shingle Weekly Reports", layout="wide", initial_sidebar_state="expanded")

# ---------- tiny helper for Streamlit reruns ----------
def safe_rerun():
    try:
        st.rerun()
    except Exception:
        pass

# ---------- Simple styling ----------
st.markdown("""
<style>
.kpi-row { display:grid; grid-template-columns:repeat(7,minmax(120px,1fr)); gap:12px; }
.kpi-card { background:#ffffff10; border:1px solid rgba(120,120,120,.15); border-radius:16px; padding:16px 14px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
.kpi-label { font-size:12px; color:#6b7280; text-transform:uppercase; letter-spacing:.04em; }
.kpi-value { font-size:22px; font-weight:700; margin-top:4px; }
hr.divider { border:none; border-top:1px solid rgba(120,120,120,.2); margin:14px 0 8px 0; }
</style>
""", unsafe_allow_html=True)

# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent
RULES_FILE = APP_DIR / "status_rules.json"
OLD_ALIASES_FILE = APP_DIR / "status_aliases.json"  # legacy import

# ---------- Printable HTML helper (Harvester) ----------
def _fmt_money(x):
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return x

def build_printable_harvester_html(harvester_name: str, summary: dict, df: pd.DataFrame) -> str:
    cols_want = [
        "City","Job Name","Start Date","Status","Sales Rep","Closer (derived)","Appointment Set By",
        "Total Contract","is_sit_harv","is_sale","is_noshow","Override To Sit","Override Note"
    ]
    present = [c for c in cols_want if c in df.columns]
    table = df[present].copy()

    if "Total Contract" in table.columns:
        table["Total Contract"] = table["Total Contract"].map(_fmt_money)
    for bcol, label in [("is_sit_harv","Sit (Harvester)"), ("is_sale","Sale"), ("is_noshow","No Show")]:
        if bcol in table.columns:
            table[label] = table[bcol].map(lambda v: "✓" if bool(v) else "")
            table.drop(columns=[bcol], inplace=True)

    order = [c for c in ["City","Job Name","Start Date","Status","Sales Rep","Closer (derived)","Appointment Set By",
                         "Total Contract","Sit (Harvester)","Sale","No Show","Override To Sit","Override Note"]
             if c in table.columns]
    table = table[order]
    table_html = table.to_html(index=False, escape=False)

    now = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    def pct(n):
        try:
            return f"{round(float(n)*100):.0f}%"
        except Exception:
            return "0%"

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Harvester Report — {harvester_name}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color:#111; }}
  h1 {{ margin: 0 0 6px; }}
  .sub {{ color:#555; margin-bottom: 16px; }}
  .grid {{ display:grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 10px; margin: 14px 0 18px; }}
  .kpi {{ border:1px solid #ddd; border-radius:12px; padding:10px 12px; }}
  .kpi .lbl {{ font-size:12px; text-transform:uppercase; color:#666; letter-spacing:.04em; }}
  .kpi .val {{ font-size:20px; font-weight:700; margin-top:4px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 8px; font-size: 12px; vertical-align: top; }}
  th {{ background:#f6f6f6; text-align:left; }}
  @media print {{ .noprint {{ display:none; }} body {{ margin: 0.25in; }} }}
</style>
</head>
<body onload="window.print()">
  <div class="noprint" style="text-align:right;margin-bottom:8px;">
    <button onclick="window.print()" style="padding:6px 10px;border-radius:8px;border:1px solid #999;background:#fff;cursor:pointer;">Print</button>
  </div>
  <h1>Harvester: {harvester_name}</h1>
  <div class="sub">Generated {now}</div>
  <div class="grid">
    <div class="kpi"><div class="lbl">Appointments</div><div class="val">{summary.get('appointments',0):,}</div></div>
    <div class="kpi"><div class="lbl">Sits (Harvester)</div><div class="val">{summary.get('sits_harv',0):,}</div></div>
    <div class="kpi"><div class="lbl">Sit %</div><div class="val">{pct(summary.get('sit_rate_harv',0.0))}</div></div>
    <div class="kpi"><div class="lbl">Sales</div><div class="val">{summary.get('sales',0):,}</div></div>
    <div class="kpi"><div class="lbl">Close %</div><div class="val">{pct(summary.get('close_rate',0.0))}</div></div>
    <div class="kpi"><div class="lbl">Sales $</div><div class="val">{_fmt_money(summary.get('sales_amt',0))}</div></div>
  </div>
  {table_html}
</body>
</html>
"""
    return html

# ---------- Numeric helpers ----------
def _coerce_money(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[^\d\.\-\(\)]", "", regex=True)
    s = s.apply(lambda x: f"-{x[1:-1]}" if re.match(r"^\(.*\)$", x) else x)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def _coerce_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)

def validate_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for money_col in ["Total Contract","Insulation Cost","Radiant Barrier Cost"]:
        if money_col in df.columns:
            df[money_col] = _coerce_money(df[money_col])
    for num_col in ["Insulation Sqft","Radiant Barrier Sqft"]:
        if num_col in df.columns:
            df[num_col] = _coerce_number(df[num_col])
    return df

# ---------- Name cleaning ----------
BAD_EMPTY = {"", "nan", "none", "null", "tbd", "-", "--"}

def clean_person(value: str) -> str:
    if not isinstance(value, str): return ""
    v = re.sub(r"\s+", " ", value.strip())
    if "," in v:
        parts = [p.strip() for p in v.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            v = f"{parts[1]} {parts[0]}"
    return " ".join(p[:1].upper()+p[1:].lower() if p else "" for p in v.split(" "))

def derive_closer_column(df: pd.DataFrame, sales_col="Sales Rep", setter_col="Appointment Set By",
                         *, normalize_names=True, setter_fallback=True) -> pd.Series:
    sales = df[sales_col].astype(str).fillna("")
    setters = df[setter_col].astype(str).fillna("")
    if normalize_names:
        sales = sales.map(clean_person)
        setters = setters.map(clean_person)
    closer = sales.copy()
    if setter_fallback:
        mask_bad = closer.str.strip().str.lower().isin(BAD_EMPTY)
        closer.loc[mask_bad] = setters.loc[mask_bad]
    return closer.fillna("")

# ---------- Rules I/O ----------
DEFAULT_RULES = {"sit_harvester": [], "sit_sales": [], "sale": [], "no_show": []}

def _import_legacy_aliases_once() -> dict:
    rules = dict(DEFAULT_RULES)
    if not OLD_ALIASES_FILE.exists():
        return rules
    try:
        aliases = json.loads(OLD_ALIASES_FILE.read_text(encoding="utf-8"))
        for s in aliases.get("Sit", []): rules["sit_harvester"].append(s); rules["sit_sales"].append(s)
        for s in aliases.get("Sold", []): rules["sale"].append(s)
        for s in aliases.get("No Show", []): rules["no_show"].append(s)
    except Exception:
        pass
    for k in rules: rules[k] = sorted(list(set(rules[k])))
    return rules

def load_rules() -> dict:
    if RULES_FILE.exists():
        try:
            data = json.loads(RULES_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and all(k in DEFAULT_RULES for k in data.keys()):
                for k in DEFAULT_RULES:
                    if k not in data: data[k] = []
                    if not isinstance(data[k], list): data[k] = []
                return data
        except Exception:
            pass
    return _import_legacy_aliases_once()

def save_rules(rules: dict):
    try:
        for k in rules:
            rules[k] = sorted(list({str(v).strip() for v in rules[k] if str(v).strip()}))
        RULES_FILE.write_text(json.dumps(rules, indent=2), encoding="utf-8")
        return True, f"Saved status rules → {RULES_FILE.name}"
    except Exception as e:
        return False, f"Could not save rules: {e}"

# ---------- Overrides (Harvester-only sit flag) ----------
def apply_overrides(df: pd.DataFrame, overrides, id_col="Job Name") -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty or id_col not in df.columns or not overrides:
        return df
    out = df.copy()
    if "Override To Sit" not in out.columns: out["Override To Sit"] = False  # Harvester only
    if "Override Note" not in out.columns: out["Override Note"] = ""
    for row_id, payload in overrides.items():
        mask = out[id_col].astype(str) == str(row_id)
        if not mask.any(): continue
        if payload.get("override_to_sit"):
            out.loc[mask, "Override To Sit"] = True
        note = payload.get("note")
        if isinstance(note, str) and note.strip():
            out.loc[mask, "Override Note"] = note.strip()
    return out

# ---------- Flags (incl. Insulation/RB) ----------
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

    # Sales flag
    is_sale_from_rule = status_clean.isin(sale_set)
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

    # Insulation / RB attributes (on sold rows; not extra sales)
    insul_cost = out.get("Insulation Cost", pd.Series(0, index=out.index))
    insul_sqft = out.get("Insulation Sqft", pd.Series(0, index=out.index))
    rb_cost    = out.get("Radiant Barrier Cost", pd.Series(0, index=out.index))
    rb_sqft    = out.get("Radiant Barrier Sqft", pd.Series(0, index=out.index))

    has_insul_any = (pd.to_numeric(insul_cost, errors="coerce").fillna(0) > 0) | \
                    (pd.to_numeric(insul_sqft, errors="coerce").fillna(0) > 0)
    has_rb_any    = (pd.to_numeric(rb_cost,  errors="coerce").fillna(0) > 0) | \
                    (pd.to_numeric(rb_sqft,  errors="coerce").fillna(0) > 0)

    out["has_insul"]      = out["is_sale"] & has_insul_any
    out["has_rb"]         = out["is_sale"] & has_rb_any
    out["has_any_addon"]  = out["is_sale"] & (has_insul_any | has_rb_any)
    return out

# ---------- Totals & Reports ----------
class Totals:
    def __init__(self, **kw): self.__dict__.update(kw)

def compute_totals(flag_df: pd.DataFrame) -> Totals:
    total_appts = int(len(flag_df))
    sits_harv   = int(flag_df["is_sit_harv"].sum())
    sits_sales  = int(flag_df["is_sit_sales"].sum())
    total_sales = int(flag_df["is_sale"].sum())
    total_noshow = int(flag_df["is_noshow"].sum())
    sales_amt = float(flag_df.loc[flag_df["is_sale"], "Total Contract"].sum()) if "Total Contract" in flag_df.columns else 0.0

    sales_with_insul = int(flag_df["has_insul"].sum())
    sales_with_rb    = int(flag_df["has_rb"].sum())
    sales_with_addon = int(flag_df["has_any_addon"].sum())

    insul_cost_sum = float(flag_df.loc[flag_df["has_insul"], "Insulation Cost"].sum()) if "Insulation Cost" in flag_df.columns else 0.0
    insul_sqft_sum = float(flag_df.loc[flag_df["has_insul"], "Insulation Sqft"].sum()) if "Insulation Sqft" in flag_df.columns else 0.0
    rb_cost_sum    = float(flag_df.loc[flag_df["has_rb"], "Radiant Barrier Cost"].sum()) if "Radiant Barrier Cost" in flag_df.columns else 0.0
    rb_sqft_sum    = float(flag_df.loc[flag_df["has_rb"], "Radiant Barrier Sqft"].sum()) if "Radiant Barrier Sqft" in flag_df.columns else 0.0

    sit_rate_harv  = (sits_harv / total_appts) if total_appts else 0.0
    sit_rate_sales = (sits_sales / total_appts) if total_appts else 0.0
    close_rate     = (total_sales / sits_sales) if sits_sales else 0.0
    sales_rate_appt= (total_sales / total_appts) if total_appts else 0.0
    avg_sale       = (sales_amt / total_sales) if total_sales else 0.0

    insul_pct = (sales_with_insul / total_sales) if total_sales else 0.0
    rb_pct    = (sales_with_rb    / total_sales) if total_sales else 0.0
    addon_pct = (sales_with_addon / total_sales) if total_sales else 0.0

    return Totals(
        total_appointments=total_appts,
        sits_harv=sits_harv, sits_sales=sits_sales,
        total_sales=total_sales, total_no_shows=total_noshow,
        total_contract_amount=round(sales_amt, 2),
        sit_rate_harv=sit_rate_harv, sit_rate_sales=sit_rate_sales,
        close_rate=close_rate, sales_rate_appt=sales_rate_appt, avg_sale=round(avg_sale, 2),
        sales_with_insul=sales_with_insul, sales_with_rb=sales_with_rb, sales_with_addon=sales_with_addon,
        insul_cost_sum=insul_cost_sum, insul_sqft_sum=insul_sqft_sum, rb_cost_sum=rb_cost_sum, rb_sqft_sum=rb_sqft_sum,
        insul_pct=insul_pct, rb_pct=rb_pct, addon_pct=addon_pct,
    )

def compute_harvester_report(flag_df: pd.DataFrame, setter_col="Appointment Set By") -> pd.DataFrame:
    df = flag_df.copy()
    if setter_col not in df.columns: df[setter_col] = ""
    rep = df.groupby(setter_col, dropna=False).apply(
        lambda g: pd.Series({"Appointments Set": len(g), "Sits": int(g["is_sit_harv"].sum())})
    ).reset_index().rename(columns={setter_col: "Harvester"})
    rep["Sit %"] = (rep["Sits"] / rep["Appointments Set"]).fillna(0.0)
    return rep

def compute_sales_report(flag_df: pd.DataFrame, closer_col: str) -> pd.DataFrame:
    df = flag_df.copy()
    if closer_col not in df.columns: df[closer_col] = ""
    def row(g: pd.DataFrame):
        appts = len(g)
        sits  = int(g["is_sit_sales"].sum())
        sales = int(g["is_sale"].sum())
        sales_amt = float(g.loc[g["is_sale"], "Total Contract"].sum()) if "Total Contract" in g.columns else 0.0

        sales_with_insul = int(g["has_insul"].sum())
        sales_with_rb    = int(g["has_rb"].sum())
        sales_with_addon = int(g["has_any_addon"].sum())

        insul_cost_sum = float(g.loc[g["has_insul"], "Insulation Cost"].sum()) if "Insulation Cost" in g.columns else 0.0
        insul_sqft_sum = float(g.loc[g["has_insul"], "Insulation Sqft"].sum()) if "Insulation Sqft" in g.columns else 0.0
        rb_cost_sum    = float(g.loc[g["has_rb"], "Radiant Barrier Cost"].sum()) if "Radiant Barrier Cost" in g.columns else 0.0
        rb_sqft_sum    = float(g.loc[g["has_rb"], "Radiant Barrier Sqft"].sum()) if "Radiant Barrier Sqft" in g.columns else 0.0

        return pd.Series({
            "Appointments": appts,
            "Sits": sits,
            "Sit %": (sits / appts) if appts else 0.0,
            "Sales": sales,
            "Close %": (sales / sits) if sits else 0.0,
            "Sales $": sales_amt,
            "Avg Sale $": (sales_amt / sales) if sales else 0.0,

            "Sales with Insulation #": sales_with_insul,
            "Insul % of Sales": (sales_with_insul / sales) if sales else 0.0,
            "Insul $": insul_cost_sum,
            "Insul Sqft": insul_sqft_sum,

            "Sales with RB #": sales_with_rb,
            "RB % of Sales": (sales_with_rb / sales) if sales else 0.0,
            "RB $": rb_cost_sum,
            "RB Sqft": rb_sqft_sum,

            "Sales with Add-on #": sales_with_addon,
            "Add-on % of Sales": (sales_with_addon / sales) if sales else 0.0,
        })
    rep = df.groupby(closer_col, dropna=False).apply(row).reset_index().rename(columns={closer_col: "Sales Rep"})
    return rep

def compute_source_report(flag_df: pd.DataFrame, source_col: str = "Source", sit_mode: str = "sales") -> pd.DataFrame:
    """
    sit_mode: 'sales' uses Sales sits (is_sit_sales) to align with Close %,
              'harvester' uses Harvester sits (is_sit_harv).
    """
    df = flag_df.copy()
    if source_col not in df.columns:
        df[source_col] = ""
    sit_flag = "is_sit_sales" if sit_mode == "sales" else "is_sit_harv"

    def row(g: pd.DataFrame):
        appts = len(g)
        sits  = int(g[sit_flag].sum())
        sales = int(g["is_sale"].sum()) if "is_sale" in g.columns else 0
        sales_amt = float(g.loc[g.get("is_sale", pd.Series(False)).astype(bool), "Total Contract"].sum()) if "Total Contract" in g.columns else 0.0

        sales_with_insul = int(g["has_insul"].sum()) if "has_insul" in g.columns else 0
        sales_with_rb    = int(g["has_rb"].sum())    if "has_rb"    in g.columns else 0
        sales_with_addon = int(g["has_any_addon"].sum()) if "has_any_addon" in g.columns else 0

        insul_cost_sum = float(g.loc[g.get("has_insul", pd.Series(False)).astype(bool), "Insulation Cost"].sum()) if "Insulation Cost" in g.columns else 0.0
        rb_cost_sum    = float(g.loc[g.get("has_rb",    pd.Series(False)).astype(bool), "Radiant Barrier Cost"].sum()) if "Radiant Barrier Cost" in g.columns else 0.0

        return pd.Series({
            "Appointments": appts,
            "Sits": sits,
            "Sit %": (sits / appts) if appts else 0.0,
            "Sales": sales,
            "Close %": (sales / sits) if sits else 0.0,
            "Sales $": sales_amt,
            "Avg Sale $": (sales_amt / sales) if sales else 0.0,

            "Sales with Insulation #": sales_with_insul,
            "Insul % of Sales": (sales_with_insul / sales) if sales else 0.0,
            "Insul $": insul_cost_sum,

            "Sales with RB #": sales_with_rb,
            "RB % of Sales": (sales_with_rb / sales) if sales else 0.0,
            "RB $": rb_cost_sum,

            "Sales with Add-on #": sales_with_addon,
            "Add-on % of Sales": (sales_with_addon / sales) if sales else 0.0,
        })

    rep = df.groupby(source_col, dropna=False).apply(row).reset_index().rename(columns={source_col: "Source"})
    return rep

def company_totals_row(flag_df: pd.DataFrame):
    t = compute_totals(flag_df)
    sales_row = {
        "Sales Rep": "TOTAL",
        "Appointments": t.total_appointments,
        "Sits": t.sits_sales,
        "Sit %": t.sit_rate_sales,
        "Sales": t.total_sales,
        "Close %": t.close_rate,
        "Sales $": t.total_contract_amount,
        "Avg Sale $": t.avg_sale,

        "Sales with Insulation #": t.sales_with_insul,
        "Insul % of Sales": t.insul_pct,
        "Insul $": t.insul_cost_sum,
        "Insul Sqft": t.insul_sqft_sum,

        "Sales with RB #": t.sales_with_rb,
        "RB % of Sales": t.rb_pct,
        "RB $": t.rb_cost_sum,
        "RB Sqft": t.rb_sqft_sum,

        "Sales with Add-on #": t.sales_with_addon,
        "Add-on % of Sales": t.addon_pct,
    }
    harv_row = {
        "Harvester": "TOTAL",
        "Appointments Set": t.total_appointments,
        "Sits": t.sits_harv,
        "Sit %": t.sit_rate_harv,
    }
    return harv_row, sales_row

# ---------- helpers ----------
def to_excel_bytes(**dfs):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df_ in dfs.items():
            df_.to_excel(writer, index=False, sheet_name=name[:31])
    return output.getvalue()

def pct_to_int(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = (out[c] * 100).round(0).astype("Int64")
    return out

def filter_detail(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "Only Sits (Harvester)": return df[df["is_sit_harv"]]
    if mode == "Only Sits (Sales)": return df[df["is_sit_sales"]]
    if mode == "Only Sales": return df[df["is_sale"]]
    if mode == "Only No Shows": return df[df["is_noshow"]]
    return df

def safe_unique_sorted(series: pd.Series, *, exclude=("TOTAL", ""), casefold=True):
    if series is None: return []
    vals = series.dropna().map(lambda x: str(x).strip()).tolist()
    vals = [v for v in vals if v not in exclude]
    vals = list(set(vals))
    return sorted(vals, key=lambda s: s.casefold()) if casefold else sorted(vals)

# ============================ UI ============================
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload Weekly CSV/Excel", type=["csv","xlsx","xls"])
if uploaded is None:
    st.info("Upload a CSV/XLSX to begin.")
    st.stop()

df_in = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)

# Column Mapper
st.sidebar.subheader("Column Mapper")
cols = list(df_in.columns)

def guess(colnames, candidates):
    for c in candidates:
        for name in colnames:
            if name.strip().lower() == c.strip().lower():
                return name
    return None

def sb_select_required(label, candidates):
    g = guess(cols, candidates)
    idx = cols.index(g) if g in cols else 0
    return st.sidebar.selectbox(label, options=cols, index=idx)

def sb_select_optional(label, candidates):
    g = guess(cols, candidates)
    opts = ["<None>"] + cols
    idx = opts.index(g) if g in opts else 0
    val = st.sidebar.selectbox(label, options=opts, index=idx)
    return None if val == "<None>" else val

job_col    = sb_select_required("Job/Record ID column", ["Job Name","Name","Job","ID","Record ID","Opportunity Name"])
status_col = sb_select_required("Status column",         ["Status","Result","Outcome"])
rep_col    = sb_select_required("Sales Rep column",      ["Sales Rep","Rep","Closer","Salesperson"])
harv_col   = sb_select_required("Appointment Set By column", ["Appointment Set By","Setter","Set By","Harvester","Created By"])
amt_col    = sb_select_required("Total Contract $ column",   ["Total Contract","Approved Estimates (Total)","Sale Amount","Contract Total","Amount"])
source_col = sb_select_optional("Source column (optional)",  ["Source","Lead Source","Campaign","Channel"])

insul_cost_col = sb_select_optional("Insulation Cost (optional)", ["Insulation Cost","Insualtion Cost","Insulation $","Insulation Amount"])
insul_sqft_col = sb_select_optional("Insulation Sqft (optional)", ["Insulation Sqft","Insualtion Sqft","Insulation SF","Insulation Square Feet"])
rb_cost_col    = sb_select_optional("Radiant Barrier Cost (optional)", ["Radiant Barrier Cost","RB Cost","Radiant Barrier $","Radiant Barrier Amount"])
rb_sqft_col    = sb_select_optional("Radiant Barrier Sqft (optional)", ["Radiant Barrier Sqft","RB Sqft","Radiant Barrier SF","Radiant Barrier Square Feet"])

# Rename -> standard internal names
raw_df = df_in.rename(columns={
    job_col: "Job Name",
    status_col: "Status",
    rep_col: "Sales Rep",
    harv_col: "Appointment Set By",
    amt_col: "Total Contract",
})
if source_col:     raw_df = raw_df.rename(columns={source_col: "Source"})
if insul_cost_col: raw_df = raw_df.rename(columns={insul_cost_col: "Insulation Cost"})
if insul_sqft_col: raw_df = raw_df.rename(columns={insul_sqft_col: "Insulation Sqft"})
if rb_cost_col:    raw_df = raw_df.rename(columns={rb_cost_col: "Radiant Barrier Cost"})
if rb_sqft_col:    raw_df = raw_df.rename(columns={rb_sqft_col: "Radiant Barrier Sqft"})

# Ensure optional columns exist
for missing, default in [
    ("Source", ""),
    ("Insulation Cost", 0.0), ("Insulation Sqft", 0.0),
    ("Radiant Barrier Cost", 0.0), ("Radiant Barrier Sqft", 0.0),
]:
    if missing not in raw_df.columns: raw_df[missing] = default

raw_df = validate_numeric(raw_df)

# ---------- Status Rules Editor (ONLY statuses from current file) ----------
st.sidebar.subheader("Status Rules (this file)")
rules = load_rules()

present_statuses = sorted(set(raw_df["Status"].dropna().astype(str).str.strip()))
rules_df = pd.DataFrame({
    "Status": present_statuses,
    "Sit (Harvester)": [s in rules.get("sit_harvester", []) for s in present_statuses],
    "Sit (Sales)":     [s in rules.get("sit_sales", []) for s in present_statuses],
    "Sale":            [s in rules.get("sale", []) for s in present_statuses],
    "No Show":         [s in rules.get("no_show", []) for s in present_statuses],
})

def merge_save_rules_from_df(existing: dict, edited_df: pd.DataFrame):
    sets = {k: set(existing.get(k, [])) for k in DEFAULT_RULES.keys()}
    for _, row in edited_df.iterrows():
        s = str(row["Status"]).strip()
        for col, key in [
            ("Sit (Harvester)", "sit_harvester"),
            ("Sit (Sales)",     "sit_sales"),
            ("Sale",            "sale"),
            ("No Show",         "no_show"),
        ]:
            if bool(row[col]): sets[key].add(s)
            else: sets[key].discard(s)
    merged = {k: sorted(list(v)) for k, v in sets.items()}
    return save_rules(merged), merged

with st.sidebar.expander("Click to review & edit status rules", expanded=True):
    st.caption("Tick the checkboxes for statuses in THIS upload. Changes auto-save & merge into your master rules.")
    edited = st.data_editor(
        rules_df,
        key="rules_editor",
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Sit (Harvester)": st.column_config.CheckboxColumn(),
            "Sit (Sales)":     st.column_config.CheckboxColumn(),
            "Sale":            st.column_config.CheckboxColumn(),
            "No Show":         st.column_config.CheckboxColumn(),
        }
    )
    if not edited.equals(rules_df):
        (ok, msg), rules = merge_save_rules_from_df(rules, edited)
        (st.success if ok else st.error)(msg)
        if ok: safe_rerun()

    st.markdown("---")
    st.markdown("**Quick Classify (one of the present statuses)**")
    pick = st.selectbox("Choose a Status", present_statuses, key="status_pick")
    cur = {
        "harv": pick in rules.get("sit_harvester", []),
        "sales": pick in rules.get("sit_sales", []),
        "sale": pick in rules.get("sale", []),
        "no_show": pick in rules.get("no_show", []),
    }
    c1, c2 = st.columns(2)
    with c1:
        f_harv = st.checkbox("Sit (Harvester)", value=cur["harv"], key="q_harv")
        f_sales = st.checkbox("Sit (Sales)", value=cur["sales"], key="q_sales")
    with c2:
        f_sale = st.checkbox("Sale", value=cur["sale"], key="q_sale")
        f_ns = st.checkbox("No Show", value=cur["no_show"], key="q_ns")
    if st.button("Save this status"):
        tmp_df = pd.DataFrame([{
            "Status": pick,
            "Sit (Harvester)": f_harv,
            "Sit (Sales)": f_sales,
            "Sale": f_sale,
            "No Show": f_ns,
        }])
        (ok, msg), rules = merge_save_rules_from_df(rules, tmp_df)
        (st.success if ok else st.error)(msg)
        if ok: safe_rerun()

# Warn about unclassified among present statuses
check_df = pd.DataFrame({
    "Status": present_statuses,
    "Sit (Harvester)": [s in rules.get("sit_harvester", []) for s in present_statuses],
    "Sit (Sales)":     [s in rules.get("sit_sales", []) for s in present_statuses],
    "Sale":            [s in rules.get("sale", []) for s in present_statuses],
    "No Show":         [s in rules.get("no_show", []) for s in present_statuses],
})
unclassified = check_df.loc[
    ~(check_df["Sit (Harvester)"] | check_df["Sit (Sales)"] | check_df["Sale"] | check_df["No Show"]),
    "Status"
].tolist()
if unclassified:
    st.warning(
        "Unclassified statuses in this file ({}): {}. "
        "Classify them above so metrics are accurate."
        .format(len(unclassified), ", ".join(unclassified))
    )

# ---------- Closer derive & ignore reps ----------
st.sidebar.subheader("Closer Name Settings")
normalize_names = st.sidebar.checkbox("Normalize names (Title Case / 'Last, First'→'First Last')", value=True)
setter_fallback = st.sidebar.checkbox("Use Appointment Set By when Sales Rep blank/TBD", value=True)
raw_df["Closer (derived)"] = derive_closer_column(raw_df, "Sales Rep", "Appointment Set By",
                                                  normalize_names=normalize_names, setter_fallback=setter_fallback)

st.sidebar.subheader("Ignore Sales Reps")
closer_options = safe_unique_sorted(raw_df["Closer (derived)"], exclude=(""," "))
ignored_closers = st.sidebar.multiselect("Exclude these closers from all reports", options=closer_options, default=[])
work_df = raw_df[~raw_df["Closer (derived)"].astype(str).isin(set(ignored_closers))].copy()

# ---------- Sales logic ----------
st.sidebar.subheader("Sales Counting Options")
infer_from_amount = st.sidebar.checkbox("Infer Sold if $ > 0", value=True)
count_sold_as_sit_harv  = st.sidebar.checkbox("Count Sold rows as Sits (Harvester)", value=True)
count_sold_as_sit_sales = st.sidebar.checkbox("Count Sold rows as Sits (Sales)", value=True)

# ---------- Sources report options ----------
st.sidebar.subheader("Sources Report Options")
source_sit_mode = st.sidebar.radio("Source Sit uses:", ["Sales sits (recommended)", "Harvester sits"], index=0)
sit_mode_key = "sales" if source_sit_mode.startswith("Sales") else "harvester"

# ---------- Overrides ----------
ROW_ID_COL = "Job Name"
if "overrides" not in st.session_state: st.session_state.overrides = {}
st.sidebar.header("Overrides")
row_ids = safe_unique_sorted(work_df["Job Name"], exclude=("",))
if row_ids:
    sel = st.sidebar.selectbox("Select a record", options=row_ids, key="override_select")
    cur_ov = st.session_state.overrides.get(sel, {"override_to_sit": False, "note": ""})
    ov_sit = st.sidebar.checkbox("Override to **HARVESTER** Sit (does NOT affect Sales)", value=cur_ov.get("override_to_sit", False))
    note = st.sidebar.text_area("Note (why?)", value=cur_ov.get("note", ""), height=80)
    a,b = st.sidebar.columns(2)
    with a:
        if st.button("Save Override"):
            st.session_state.overrides[sel] = {"override_to_sit": ov_sit, "note": note}
            safe_rerun()
    with b:
        if st.button("Clear Override"):
            if sel in st.session_state.overrides: del st.session_state.overrides[sel]
            safe_rerun()

# Apply overrides & build flags
work_df = apply_overrides(work_df, st.session_state.overrides, id_col=ROW_ID_COL)
flag_df = build_flags(
    work_df,
    rules=rules,
    infer_sale_from_amount=infer_from_amount,
    count_sold_as_sit_harv=count_sold_as_sit_harv,
    count_sold_as_sit_sales=count_sold_as_sit_sales
)

# ---------- Totals & Reports ----------
totals = compute_totals(flag_df)
harvester_report = compute_harvester_report(flag_df, setter_col="Appointment Set By")
sales_report     = compute_sales_report(flag_df, closer_col="Closer (derived)")
source_report    = compute_source_report(flag_df, source_col="Source", sit_mode=sit_mode_key)

# Append TOTAL rows
harv_total_row, sales_total_row = company_totals_row(flag_df)
harvester_with_total = pd.concat([harvester_report, pd.DataFrame([harv_total_row])], ignore_index=True)
sales_with_total     = pd.concat([sales_report,     pd.DataFrame([sales_total_row])], ignore_index=True)

src_tot = compute_totals(flag_df)
source_total_row = {
    "Source": "TOTAL",
    "Appointments": src_tot.total_appointments,
    "Sits": src_tot.sits_sales if sit_mode_key == "sales" else src_tot.sits_harv,
    "Sit %": src_tot.sit_rate_sales if sit_mode_key == "sales" else src_tot.sit_rate_harv,
    "Sales": src_tot.total_sales,
    "Close %": src_tot.close_rate,
    "Sales $": src_tot.total_contract_amount,
    "Avg Sale $": src_tot.avg_sale,
    "Sales with Insulation #": src_tot.sales_with_insul,
    "Insul % of Sales": src_tot.insul_pct,
    "Insul $": src_tot.insul_cost_sum,
    "Sales with RB #": src_tot.sales_with_rb,
    "RB % of Sales": src_tot.rb_pct,
    "RB $": src_tot.rb_cost_sum,
    "Sales with Add-on #": src_tot.sales_with_addon,
    "Add-on % of Sales": src_tot.addon_pct,
}
source_with_total = pd.concat([source_report, pd.DataFrame([source_total_row])], ignore_index=True)

# Display-friendly % (whole numbers)
harv_display   = pct_to_int(harvester_with_total, ["Sit %"])
sales_display  = pct_to_int(sales_with_total, ["Sit %","Close %","Insul % of Sales","RB % of Sales","Add-on % of Sales"])
source_display = pct_to_int(source_with_total,    ["Sit %","Close %","Insul % of Sales","RB % of Sales","Add-on % of Sales"])

# Company Total (single-row)
company_total_table = pd.DataFrame([{
    "Appointments": totals.total_appointments,
    "Sits (Harvester)": totals.sits_harv,
    "Sit % (Harvester)": int(round(totals.sit_rate_harv * 100, 0)),
    "Sits (Sales)": totals.sits_sales,
    "Sit % (Sales)": int(round(totals.sit_rate_sales * 100, 0)),
    "Sales": totals.total_sales,
    "Close %": int(round(totals.close_rate * 100, 0)),
    "Sales $": totals.total_contract_amount,
    "Avg Sale $": totals.avg_sale,
    "Sales with Insulation #": totals.sales_with_insul,
    "Insul % of Sales": int(round(totals.insul_pct * 100, 0)),
    "Insul $": totals.insul_cost_sum,
    "Insul Sqft": totals.insul_sqft_sum,
    "Sales with RB #": totals.sales_with_rb,
    "RB % of Sales": int(round(totals.rb_pct * 100, 0)),
    "RB $": totals.rb_cost_sum,
    "RB Sqft": totals.rb_sqft_sum,
    "Sales with Add-on #": totals.sales_with_addon,
    "Add-on % of Sales": int(round(totals.addon_pct * 100, 0)),
}])

# Detail view
DETAIL_COLS = [
    "City","Job Name","Start Date","Status","Source","Sales Rep","Closer (derived)","Appointment Set By",
    "Total Contract","Insulation Cost","Insulation Sqft","Radiant Barrier Cost","Radiant Barrier Sqft",
    "is_sit_harv","is_sit_sales","is_sale","is_noshow","has_insul","has_rb","has_any_addon",
    "Override To Sit","Override Note"
]
detail_display = flag_df.copy()
for col in ["is_sit_harv","is_sit_sales","is_sale","is_noshow","has_insul","has_rb","has_any_addon"]:
    if col in detail_display.columns:
        detail_display[col] = detail_display[col].astype(bool)

# ---------- Header & KPI cards ----------
st.title("US Shingle Weekly Reports")
st.caption("Tip: ⌘/Ctrl + B toggles the sidebar.")
if ignored_closers:
    st.info(f"Ignoring {len(ignored_closers)} closer(s): {', '.join(ignored_closers)}")

st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
kpis = [
    ("Appointments", f"{totals.total_appointments:,}"),
    ("Sits (Harvester)", f"{totals.sits_harv:,}"),
    ("Sit % (Harvester)", f"{totals.sit_rate_harv:.0%}"),
    ("Sales", f"{totals.total_sales:,}"),
    ("Close %", f"{totals.close_rate:.0%}"),
    ("Sales $", f"${totals.total_contract_amount:,.0f}"),
    ("Avg Sale $", f"${totals.avg_sale:,.0f}"),
]
for label, value in kpis:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ---------- Tabs ----------
tab_overview, tab_sources, tab_setters, tab_closers, tab_detail, tab_audit, tab_exports = st.tabs(
    ["📊 Overview", "📣 Sources", "🌱 Setters (Harvester)", "💼 Closers (Sales)", "🧾 Detail", "📝 Audit", "⬇️ Export"]
)

with tab_overview:
    st.subheader("🏢 Company Total (All)")
    st.dataframe(
        company_total_table,
        use_container_width=True, hide_index=True,
        column_config={
            "Sit % (Harvester)": st.column_config.NumberColumn(format="%d%%"),
            "Sit % (Sales)":     st.column_config.NumberColumn(format="%d%%"),
            "Close %":           st.column_config.NumberColumn(format="%d%%"),
            "Sales $":           st.column_config.NumberColumn(format="$%.0f"),
            "Avg Sale $":        st.column_config.NumberColumn(format="$%.0f"),
            "Insul % of Sales":  st.column_config.NumberColumn(format="%d%%"),
            "RB % of Sales":     st.column_config.NumberColumn(format="%d%%"),
            "Add-on % of Sales": st.column_config.NumberColumn(format="%d%%"),
            "Insul $":           st.column_config.NumberColumn(format="$%.0f"),
            "RB $":              st.column_config.NumberColumn(format="$%.0f"),
        }
    )

    st.subheader("Quick Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Harvester Snapshot**")
        st.dataframe(
            harv_display, use_container_width=True, hide_index=True,
            column_config={"Sit %": st.column_config.NumberColumn(format="%d%%")}
        )
    with c2:
        st.markdown("**Sales Snapshot (incl. Insulation / RB)**")
        st.dataframe(
            sales_display, use_container_width=True, hide_index=True,
            column_config={
                "Sit %":   st.column_config.NumberColumn(format="%d%%"),
                "Close %": st.column_config.NumberColumn(format="%d%%"),
                "Insul % of Sales": st.column_config.NumberColumn(format="%d%%"),
                "RB % of Sales":    st.column_config.NumberColumn(format="%d%%"),
                "Add-on % of Sales":st.column_config.NumberColumn(format="%d%%"),
                "Sales $": st.column_config.NumberColumn(format="$%.0f")),
        )

with tab_sources:
    st.subheader("Sources Report")
    st.caption(f"Sits use: **{'Sales sits' if sit_mode_key=='sales' else 'Harvester sits'}**")
    st.dataframe(
        source_display, use_container_width=True, hide_index=True,
        column_config={
            "Sit %":   st.column_config.NumberColumn(format="%d%%"),
            "Close %": st.column_config.NumberColumn(format="%d%%"),
            "Insul % of Sales": st.column_config.NumberColumn(format="%d%%"),
            "RB % of Sales":    st.column_config.NumberColumn(format="%d%%"),
            "Add-on % of Sales":st.column_config.NumberColumn(format="%d%%"),
            "Sales $": st.column_config.NumberColumn(format="$%.0f"),
            "Avg Sale $": st.column_config.NumberColumn(format="$%.0f"),
            "Insul $": st.column_config.NumberColumn(format="$%.0f"),
            "RB $":    st.column_config.NumberColumn(format="$%.0f"),
        }
    )

    st.markdown("### Raw Line Items for a Source")
    src_list = safe_unique_sorted(source_report["Source"])
    if src_list:
        sel_s = st.selectbox("Choose Source", options=src_list, key="source_choice")
        only_sits_label = "Only Sits (Sales)" if sit_mode_key == "sales" else "Only Sits (Harvester)"
        mode_s = st.radio("Filter:", ["All", only_sits_label, "Only Sales", "Only No Shows"],
                          horizontal=True, key="src_mode")

        s_detail = detail_display[ detail_display["Source"].astype(str).str.strip() == sel_s ].copy()
        if mode_s == "Only Sales":
            s_detail = s_detail[s_detail["is_sale"]]
        elif mode_s == "Only No Shows":
            s_detail = s_detail[s_detail["is_noshow"]]
        elif mode_s != "All":
            s_detail = s_detail[s_detail["is_sit_sales"] if sit_mode_key=="sales" else s_detail["is_sit_harv"]]

        st.dataframe(s_detail[[c for c in DETAIL_COLS if c in s_detail.columns]], use_container_width=True, hide_index=True)
        st.download_button(
            f"Download Raw CSV — {sel_s}",
            data=s_detail.to_csv(index=False).encode("utf-8"),
            file_name=f"source_raw_{sel_s.replace(' ','_')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No sources found.")

with tab_setters:
    st.subheader("Harvester Report (Appointment Set By)")
    st.dataframe(harv_display, use_container_width=True, hide_index=True,
                 column_config={"Sit %": st.column_config.NumberColumn(format="%d%%")})

    st.markdown("### Raw Line Items for a Harvester")
    harv_list = safe_unique_sorted(harvester_report["Harvester"])
    if harv_list:
        sel_h = st.selectbox("Choose Harvester", options=harv_list, key="harv_choice")
        mode_h = st.radio("Filter:", ["All", "Only Sits (Harvester)", "Only Sales", "Only No Shows"],
                          horizontal=True, key="harv_mode")
        h_all = detail_display[ detail_display["Appointment Set By"].astype(str).str.strip() == sel_h ].copy()
        h_detail = filter_detail(h_all, mode_h)
        st.dataframe(h_detail[[c for c in DETAIL_COLS if c in h_detail.columns]],
                     use_container_width=True, hide_index=True)

        # Summary for selected harvester (ALL rows for that harvester)
        try:
            apps = int(len(h_all))
            sits_h = int(h_all["is_sit_harv"].sum()) if "is_sit_harv" in h_all.columns else 0
            sits_s = int(h_all["is_sit_sales"].sum()) if "is_sit_sales" in h_all.columns else 0
            sales_n = int(h_all["is_sale"].sum()) if "is_sale" in h_all.columns else 0
            sales_amt = float(h_all.loc[h_all.get("is_sale", pd.Series(False)).astype(bool), "Total Contract"].sum()) if "Total Contract" in h_all.columns else 0.0
            sit_rate_h = (sits_h / apps) if apps else 0.0
            close_rate = (sales_n / sits_s) if sits_s else 0.0
        except Exception:
            apps=sits_h=sits_s=sales_n=0; sales_amt=0.0; sit_rate_h=close_rate=0.0

        summary = {
            "appointments": apps,
            "sits_harv": sits_h,
            "sit_rate_harv": sit_rate_h,
            "sales": sales_n,
            "close_rate": close_rate,
            "sales_amt": sales_amt,
        }

        # Printable HTML for this harvester (uses current filtered rows)
        html_doc = build_printable_harvester_html(sel_h, summary, h_detail)

        cA, cB = st.columns(2)
        with cA:
            if st.button("🖨 Print This Harvester (with overrides & notes)", key="print_harvester"):
                js = f"""
                <script>
                  const html = `{html_doc.replace("\\", "\\\\").replace("`", "\\`")}`;
                  const w = window.open("", "_blank");
                  w.document.open();
                  w.document.write(html);
                  w.document.close();
                </script>
                """
                components.html(js, height=0)
        with cB:
            st.download_button(
                "⬇️ Download Printable HTML",
                data=html_doc.encode("utf-8"),
                file_name=f"harvester_{sel_h.replace(' ','_')}.html",
                mime="text/html"
            )

        st.download_button(
            f"Download Raw CSV — {sel_h}",
            data=h_detail.to_csv(index=False).encode("utf-8"),
            file_name=f"harvester_raw_{sel_h.replace(' ','_')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No harvesters found.")

with tab_closers:
    st.subheader("Sales Report (Sales Rep — using **Closer (derived)**)")
    st.dataframe(
        sales_display, use_container_width=True, hide_index=True,
        column_config={
            "Sit %":   st.column_config.NumberColumn(format="%d%%"),
            "Close %": st.column_config.NumberColumn(format="%d%%"),
            "Insul % of Sales": st.column_config.NumberColumn(format="%d%%"),
            "RB % of Sales":    st.column_config.NumberColumn(format="%d%%"),
            "Add-on % of Sales":st.column_config.NumberColumn(format="%d%%"),
            "Sales $": st.column_config.NumberColumn(format="$%.0f"),
            "Avg Sale $": st.column_config.NumberColumn(format("$%.0f")),
            "Insul $": st.column_config.NumberColumn(format("$%.0f")),
            "RB $":    st.column_config.NumberColumn(format("$%.0f")),
        }
    )

    st.markdown("### Raw Line Items for a Sales Rep")
    rep_list = safe_unique_sorted(sales_report["Sales Rep"])
    if rep_list:
        sel_r = st.selectbox("Choose Sales Rep", options=rep_list, key="rep_choice")
        mode_r = st.radio("Filter:", ["All", "Only Sits (Sales)", "Only Sales", "Only No Shows"],
                          horizontal=True, key="rep_mode")
        r_detail = detail_display[ detail_display["Closer (derived)"].astype(str).str.strip() == sel_r ].copy()
        r_detail = filter_detail(r_detail, mode_r)
        st.dataframe(r_detail[[c for c in DETAIL_COLS if c in r_detail.columns]], use_container_width=True, hide_index=True)
        st.download_button(
            f"Download Raw CSV — {sel_r}",
            data=r_detail.to_csv(index=False).encode("utf-8"),
            file_name=f"sales_raw_{sel_r.replace(' ','_')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No sales reps found.")

with tab_detail:
    st.subheader("Detail (with flags)")
    st.dataframe(detail_display, use_container_width=True, hide_index=True)

with tab_audit:
    if st.session_state.overrides:
        st.subheader("Overrides & Notes (Audit)")
        audit_rows = []
        for row_id, p in st.session_state.overrides.items():
            audit_rows.append({"Job Name": row_id, "Override To Sit": bool(p.get("override_to_sit")), "Note": p.get("note","")})
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No overrides yet.")

with tab_exports:
    st.subheader("Export Reports")
    colA, colB = st.columns(2)
    with colA:
        st.download_button("Download Harvester CSV", data=harv_display.to_csv(index=False).encode("utf-8"),
                           file_name="harvester_report.csv", mime="text/csv")
        st.download_button("Download Sales CSV", data=sales_display.to_csv(index=False).encode("utf-8"),
                           file_name="sales_report.csv", mime="text/csv")
        st.download_button("Download Sources CSV", data=source_display.to_csv(index=False).encode("utf-8"),
                           file_name="sources_report.csv", mime="text/csv")
        st.download_button("Download Company Total CSV", data=company_total_table.to_csv(index=False).encode("utf-8"),
                           file_name="company_total.csv", mime="text/csv")
    with colB:
        all_bytes = to_excel_bytes(
            CompanyTotal=company_total_table,
            Harvester=harv_display,
            Sales=sales_display,
            Sources=source_display,
            Detail=detail_display
        )
        st.download_button("Download Excel (All Sheets)", data=all_bytes,
                           file_name="weekly_reports.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Optional general page print (kept for convenience)
st.write("")
if st.button("Print Page"):
    components.html("<script>window.print();</script>", height=0)
