# app.py — US Shingle Weekly Reports (with Insulation / Radiant Barrier analytics)

import json, re
from io import BytesIO
from pathlib import Path

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
.kpi-label { font-size:12px; color:var(--text-color,#6b7280); text-transform:uppercase; letter-spacing:.04em; }
.kpi-value { font-size:22px; font-weight:700; margin-top:4px; }
hr.divider { border:none; border-top:1px solid rgba(120,120,120,.2); margin:14px 0 8px 0; }
</style>
""", unsafe_allow_html=True)

# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent
ALIASES_FILE = APP_DIR / "status_aliases.json"

# ---------- Base normalization ----------
BASE_NORMALIZATION = {
    "sit": "Sit", "sits": "Sit", "seated": "Sit",
    "sold": "Sold", "sale": "Sold", "sold - finance": "Sold",
    "closed": "Sold", "closed won": "Sold", "signed": "Sold",
    "contract": "Sold", "contract signed": "Sold", "agreement signed": "Sold",
    "install": "Sold", "installed": "Sold", "booked": "Sold", "signed contract": "Sold",
    "no show": "No Show", "noshow": "No Show", "no-show": "No Show", "no show- h/o": "No Show",
    "set": "Set", "appt set": "Set", "appointment set": "Set",
    "cancel": "Cancel", "canceled": "Cancel", "cancelled": "Cancel",
}
DEFAULT_EXTRA_ALIASES = {
    "Sit":  ["Sit - No Sale", "Sit - Pending", "Sit - Sold"],
    "Sold": ["Signed Contract", "Sit - Sold"],
    "No Show": ["No Show- H/O", "NS", "No-Show", "NoShow", "No Sit"],
    "Set": [],
}

def load_aliases() -> dict:
    try:
        if ALIASES_FILE.exists():
            data = json.loads(ALIASES_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and all(k in ["Sit","Sold","No Show","Set"] for k in data.keys()):
                return data
    except Exception:
        pass
    return DEFAULT_EXTRA_ALIASES.copy()

def save_aliases(extra_aliases: dict):
    try:
        ALIASES_FILE.write_text(json.dumps(extra_aliases, indent=2), encoding="utf-8")
        return True, f"Saved defaults to {ALIASES_FILE.name}"
    except Exception as e:
        return False, f"Could not save defaults: {e}"

def normalize_factory(extra_aliases: dict):
    norm = dict(BASE_NORMALIZATION)
    for label, aliases in (extra_aliases or {}).items():
        for a in aliases or []:
            a = str(a).strip().lower()
            if a: norm[a] = label
    def _normalize_status(value):
        if not isinstance(value, str): return "Unknown"
        v = value.strip()
        return norm.get(v.lower(), v)
    return _normalize_status

# ---------- Money / numeric helpers ----------
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

# ---------- Overrides ----------
def apply_overrides(df: pd.DataFrame, overrides, id_col="Job Name") -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty or id_col not in df.columns or not overrides:
        return df
    out = df.copy()
    if "Override To Sit" not in out.columns: out["Override To Sit"] = False
    if "Override Note" not in out.columns: out["Override Note"] = ""
    for row_id, payload in overrides.items():
        mask = out[id_col].astype(str) == str(row_id)
        if not mask.any(): continue
        if payload.get("override_to_sit"):
            out.loc[mask, "Status"] = "Sit"
            out.loc[mask, "Override To Sit"] = True
        note = payload.get("note")
        if isinstance(note, str) and note.strip():
            out.loc[mask, "Override Note"] = note.strip()
    return out

# ---------- Flags (incl. Insulation/RB) ----------
def build_flags(df: pd.DataFrame, *, infer_sale_from_amount: bool, count_sold_as_sit: bool) -> pd.DataFrame:
    out = df.copy()
    s = out["Status"].astype(str).str.strip().str.lower()

    sit_like  = (s.str.startswith("sit") | s.str.contains(r"\bsit\b", regex=True) | s.isin({"sit"}))
    sold_like = (s.isin({"sold","signed contract"}) | s.str.contains(r"\bsold\b", regex=True) | s.str.contains(r"signed\s*contract", regex=True))
    noshow_like = s.str.contains("no show", regex=False)

    dollars_pos = pd.Series(False, index=out.index)
    if "Total Contract" in out.columns:
        dollars_pos = pd.to_numeric(out["Total Contract"], errors="coerce").fillna(0) > 0

    out["is_sale"] = sold_like | (infer_sale_from_amount & dollars_pos)
    out["is_sit"]  = sit_like | (count_sold_as_sit & out["is_sale"])
    out["is_noshow"] = noshow_like

    if "Override To Sit" in out.columns:
        out.loc[out["Override To Sit"].fillna(False).astype(bool), "is_sit"] = True

    # Insulation / RB flags (count only on sold rows)
    insul_cost = out.get("Insulation Cost", pd.Series(0, index=out.index))
    insul_sqft = out.get("Insulation Sqft", pd.Series(0, index=out.index))
    rb_cost    = out.get("Radiant Barrier Cost", pd.Series(0, index=out.index))
    rb_sqft    = out.get("Radiant Barrier Sqft", pd.Series(0, index=out.index))

    has_insul_any = (pd.to_numeric(insul_cost, errors="coerce").fillna(0) > 0) | \
                    (pd.to_numeric(insul_sqft, errors="coerce").fillna(0) > 0)
    has_rb_any    = (pd.to_numeric(rb_cost, errors="coerce").fillna(0) > 0)   | \
                    (pd.to_numeric(rb_sqft, errors="coerce").fillna(0) > 0)

    out["has_insul"] = out["is_sale"] & has_insul_any
    out["has_rb"]    = out["is_sale"] & has_rb_any
    out["has_any_addon"] = out["is_sale"] & (has_insul_any | has_rb_any)  # counts a sale once even if both
    return out

# ---------- Totals & Reports ----------
class Totals:
    def __init__(self, **kw): self.__dict__.update(kw)

def compute_totals(flag_df: pd.DataFrame) -> Totals:
    total_appts = int(len(flag_df))
    total_sits  = int(flag_df["is_sit"].sum())
    total_sales = int(flag_df["is_sale"].sum())
    total_noshow = int(flag_df["is_noshow"].sum())

    sales_amt = float(flag_df.loc[flag_df["is_sale"], "Total Contract"].sum()) if "Total Contract" in flag_df.columns else 0.0

    # Insulation / RB sums on sold rows
    insul_sales = int(flag_df["has_insul"].sum())
    rb_sales    = int(flag_df["has_rb"].sum())
    addon_sales = int(flag_df["has_any_addon"].sum())

    insul_cost_sum = float(flag_df.loc[flag_df["has_insul"], "Insulation Cost"].sum()) if "Insulation Cost" in flag_df.columns else 0.0
    insul_sqft_sum = float(flag_df.loc[flag_df["has_insul"], "Insulation Sqft"].sum()) if "Insulation Sqft" in flag_df.columns else 0.0
    rb_cost_sum    = float(flag_df.loc[flag_df["has_rb"], "Radiant Barrier Cost"].sum()) if "Radiant Barrier Cost" in flag_df.columns else 0.0
    rb_sqft_sum    = float(flag_df.loc[flag_df["has_rb"], "Radiant Barrier Sqft"].sum()) if "Radiant Barrier Sqft" in flag_df.columns else 0.0

    sit_rate_appt   = (total_sits / total_appts) if total_appts else 0.0
    close_rate      = (total_sales / total_sits) if total_sits else 0.0
    sales_rate_appt = (total_sales / total_appts) if total_appts else 0.0
    avg_sale        = (sales_amt / total_sales) if total_sales else 0.0

    insul_pct = (insul_sales / total_sales) if total_sales else 0.0
    rb_pct    = (rb_sales / total_sales) if total_sales else 0.0
    addon_pct = (addon_sales / total_sales) if total_sales else 0.0

    return Totals(
        total_appointments=total_appts,
        total_sits=total_sits,
        total_sales=total_sales,
        total_no_shows=total_noshow,
        total_contract_amount=round(sales_amt, 2),
        sit_rate_appt=sit_rate_appt,
        close_rate=close_rate,
        sales_rate_appt=sales_rate_appt,
        avg_sale=round(avg_sale, 2),

        insul_sales=insul_sales, rb_sales=rb_sales, addon_sales=addon_sales,
        insul_cost_sum=insul_cost_sum, insul_sqft_sum=insul_sqft_sum,
        rb_cost_sum=rb_cost_sum, rb_sqft_sum=rb_sqft_sum,
        insul_pct=insul_pct, rb_pct=rb_pct, addon_pct=addon_pct,
    )

def compute_harvester_report(flag_df: pd.DataFrame, setter_col="Appointment Set By") -> pd.DataFrame:
    df = flag_df.copy()
    if setter_col not in df.columns: df[setter_col] = ""
    rep = df.groupby(setter_col, dropna=False).apply(
        lambda g: pd.Series({"Appointments Set": len(g), "Sits": int(g["is_sit"].sum())})
    ).reset_index().rename(columns={setter_col: "Harvester"})
    rep["Sit %"] = (rep["Sits"] / rep["Appointments Set"]).fillna(0.0)
    return rep

def compute_sales_report(flag_df: pd.DataFrame, closer_col: str) -> pd.DataFrame:
    df = flag_df.copy()
    if closer_col not in df.columns: df[closer_col] = ""
    def row(g: pd.DataFrame):
        appts = len(g)
        sits  = int(g["is_sit"].sum())
        sales = int(g["is_sale"].sum())
        sales_amt = float(g.loc[g["is_sale"], "Total Contract"].sum()) if "Total Contract" in g.columns else 0.0

        insul_sales = int(g["has_insul"].sum())
        rb_sales    = int(g["has_rb"].sum())
        addon_sales = int(g["has_any_addon"].sum())

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

            "Insul Sales #": insul_sales,
            "Insul % of Sales": (insul_sales / sales) if sales else 0.0,
            "Insul $": insul_cost_sum,
            "Insul Sqft": insul_sqft_sum,

            "RB Sales #": rb_sales,
            "RB % of Sales": (rb_sales / sales) if sales else 0.0,
            "RB $": rb_cost_sum,
            "RB Sqft": rb_sqft_sum,

            "Add-on Sales #": addon_sales,
            "Add-on % of Sales": (addon_sales / sales) if sales else 0.0,
        })
    rep = df.groupby(closer_col, dropna=False).apply(row).reset_index().rename(columns={closer_col: "Sales Rep"})
    return rep

def company_totals_row(flag_df: pd.DataFrame):
    t = compute_totals(flag_df)
    sales_row = {
        "Sales Rep": "TOTAL",
        "Appointments": t.total_appointments,
        "Sits": t.total_sits,
        "Sit %": t.sit_rate_appt,
        "Sales": t.total_sales,
        "Close %": t.close_rate,
        "Sales $": t.total_contract_amount,
        "Avg Sale $": t.avg_sale,

        "Insul Sales #": t.insul_sales,
        "Insul % of Sales": t.insul_pct,
        "Insul $": t.insul_cost_sum,
        "Insul Sqft": t.insul_sqft_sum,

        "RB Sales #": t.rb_sales,
        "RB % of Sales": t.rb_pct,
        "RB $": t.rb_cost_sum,
        "RB Sqft": t.rb_sqft_sum,

        "Add-on Sales #": t.addon_sales,
        "Add-on % of Sales": t.addon_pct,
    }
    harv_row = {
        "Harvester": "TOTAL",
        "Appointments Set": t.total_appointments,
        "Sits": t.total_sits,
        "Sit %": t.sit_rate_appt,
    }
    return harv_row, sales_row

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
    if mode == "Only Sits": return df[df["is_sit"]]
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

# NEW optional columns
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
if insul_cost_col: raw_df = raw_df.rename(columns={insul_cost_col: "Insulation Cost"})
if insul_sqft_col: raw_df = raw_df.rename(columns={insul_sqft_col: "Insulation Sqft"})
if rb_cost_col:    raw_df = raw_df.rename(columns={rb_cost_col: "Radiant Barrier Cost"})
if rb_sqft_col:    raw_df = raw_df.rename(columns={rb_sqft_col: "Radiant Barrier Sqft"})

# Ensure optional columns exist
for missing, default in [
    ("Insulation Cost", 0.0), ("Insulation Sqft", 0.0),
    ("Radiant Barrier Cost", 0.0), ("Radiant Barrier Sqft", 0.0),
]:
    if missing not in raw_df.columns: raw_df[missing] = default

raw_df = validate_numeric(raw_df)

# Status Aliases
st.sidebar.subheader("Status Aliases (optional, one-time)")
loaded_aliases = load_aliases()
sit_aliases_text    = st.sidebar.text_input("Aliases → Sit",     value=", ".join(loaded_aliases.get("Sit", [])))
sold_aliases_text   = st.sidebar.text_input("Aliases → Sold",    value=", ".join(loaded_aliases.get("Sold", [])))
noshow_aliases_text = st.sidebar.text_input("Aliases → No Show", value=", ".join(loaded_aliases.get("No Show", [])))
set_aliases_text    = st.sidebar.text_input("Aliases → Set",     value=", ".join(loaded_aliases.get("Set", [])))

extra_aliases = {
    "Sit":     [s.strip() for s in sit_aliases_text.split(",") if s.strip()],
    "Sold":    [s.strip() for s in sold_aliases_text.split(",") if s.strip()],
    "No Show": [s.strip() for s in noshow_aliases_text.split(",") if s.strip()],
    "Set":     [s.strip() for s in set_aliases_text.split(",") if s.strip()],
}
c1, c2, c3 = st.sidebar.columns(3)
with c1:
    if st.button("Save defaults"):
        ok, msg = save_aliases(extra_aliases)
        (st.success if ok else st.error)(msg)
with c2:
    st.download_button("Download", data=json.dumps(extra_aliases, indent=2),
                       file_name="status_aliases.json", mime="application/json")
with c3:
    up = st.file_uploader("Upload", type=["json"], label_visibility="collapsed")
    if up is not None:
        try:
            incoming = json.load(up)
            if all(k in incoming for k in ["Sit","Sold","No Show","Set"]):
                ok, msg = save_aliases(incoming)
                (st.success if ok else st.error)(msg)
                safe_rerun()
            else:
                st.error("Invalid mapping file (missing keys).")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

_normalize_status = normalize_factory(extra_aliases)
raw_df["Status"] = raw_df["Status"].apply(_normalize_status)

# Closer derive
st.sidebar.subheader("Closer Name Settings")
normalize_names = st.sidebar.checkbox("Normalize names (Title Case / 'Last, First'→'First Last')", value=True)
setter_fallback = st.sidebar.checkbox("Use Appointment Set By when Sales Rep blank/TBD", value=True)
raw_df["Closer (derived)"] = derive_closer_column(raw_df, "Sales Rep", "Appointment Set By",
                                                  normalize_names=normalize_names, setter_fallback=setter_fallback)

# Ignore Sales Reps
st.sidebar.subheader("Ignore Sales Reps")
closer_options = safe_unique_sorted(raw_df["Closer (derived)"], exclude=(""," "))
ignored_closers = st.sidebar.multiselect("Exclude these closers from all reports", options=closer_options, default=[])
work_df = raw_df[~raw_df["Closer (derived)"].astype(str).isin(set(ignored_closers))].copy()

# Sales logic
st.sidebar.subheader("Sales Counting Options")
infer_from_amount = st.sidebar.checkbox("Infer Sold if $ > 0", value=True)
count_sold_as_sit = st.sidebar.checkbox("Count Sold rows as Sits", value=True)

# Initial flags
flag_df = build_flags(work_df, infer_sale_from_amount=infer_from_amount, count_sold_as_sit=count_sold_as_sit)

# Overrides
ROW_ID_COL = "Job Name"
if "overrides" not in st.session_state: st.session_state.overrides = {}
st.sidebar.header("Overrides")
row_ids = safe_unique_sorted(work_df["Job Name"], exclude=("",))
if row_ids:
    sel = st.sidebar.selectbox("Select a record", options=row_ids, key="override_select")
    cur = st.session_state.overrides.get(sel, {"override_to_sit": False, "note": ""})
    ov_sit = st.sidebar.checkbox("Override this record to a SIT", value=cur.get("override_to_sit", False))
    note = st.sidebar.text_area("Note (why?)", value=cur.get("note", ""), height=80)
    a,b = st.sidebar.columns(2)
    with a:
        if st.button("Save Override"):
            st.session_state.overrides[sel] = {"override_to_sit": ov_sit, "note": note}
            safe_rerun()
    with b:
        if st.button("Clear Override"):
            if sel in st.session_state.overrides: del st.session_state.overrides[sel]
            safe_rerun()

# Re-apply overrides & flags
flag_df = apply_overrides(flag_df, st.session_state.overrides, id_col=ROW_ID_COL)
flag_df = build_flags(flag_df, infer_sale_from_amount=infer_from_amount, count_sold_as_sit=count_sold_as_sit)

# Compute totals & reports
totals = compute_totals(flag_df)
harvester_report = compute_harvester_report(flag_df, setter_col="Appointment Set By")
sales_report     = compute_sales_report(flag_df, closer_col="Closer (derived)")

# Append TOTAL rows
harv_total_row, sales_total_row = company_totals_row(flag_df)
harvester_with_total = pd.concat([harvester_report, pd.DataFrame([harv_total_row])], ignore_index=True)
sales_with_total     = pd.concat([sales_report,     pd.DataFrame([sales_total_row])], ignore_index=True)

# Percent display (whole numbers)
harv_display  = pct_to_int(harvester_with_total, ["Sit %"])
sales_display = pct_to_int(
    sales_with_total,
    ["Sit %","Close %","Insul % of Sales","RB % of Sales","Add-on % of Sales"]
)

# Company Total (single-row)
company_total_table = pd.DataFrame([{
    "Appointments": totals.total_appointments,
    "Sits": totals.total_sits,
    "Sit %": int(round(totals.sit_rate_appt * 100, 0)),
    "Sales": totals.total_sales,
    "Close %": int(round(totals.close_rate * 100, 0)),
    "Sales $": totals.total_contract_amount,
    "Avg Sale $": totals.avg_sale,

    "Insul Sales #": totals.insul_sales,
    "Insul % of Sales": int(round(totals.insul_pct * 100, 0)),
    "Insul $": totals.insul_cost_sum,
    "Insul Sqft": totals.insul_sqft_sum,

    "RB Sales #": totals.rb_sales,
    "RB % of Sales": int(round(totals.rb_pct * 100, 0)),
    "RB $": totals.rb_cost_sum,
    "RB Sqft": totals.rb_sqft_sum,

    "Add-on Sales #": totals.addon_sales,
    "Add-on % of Sales": int(round(totals.addon_pct * 100, 0)),
}])

# Detail view
DETAIL_COLS = [
    "City","Job Name","Start Date","Status","Sales Rep","Closer (derived)","Appointment Set By",
    "Total Contract","Insulation Cost","Insulation Sqft","Radiant Barrier Cost","Radiant Barrier Sqft",
    "is_sit","is_sale","is_noshow","has_insul","has_rb","has_any_addon"
]
detail_display = flag_df.copy()
for col in ["is_sit","is_sale","is_noshow","has_insul","has_rb","has_any_addon"]:
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
    ("Sits", f"{totals.total_sits:,}"),
    ("Sit %", f"{totals.sit_rate_appt:.0%}"),
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
tab_overview, tab_setters, tab_closers, tab_detail, tab_audit, tab_exports = st.tabs(
    ["📊 Overview", "🌱 Setters (Harvester)", "💼 Closers (Sales)", "🧾 Detail", "📝 Audit", "⬇️ Export"]
)

with tab_overview:
    st.subheader("🏢 Company Total (All)")
    st.dataframe(
        company_total_table,
        use_container_width=True, hide_index=True,
        column_config={
            "Sit %":   st.column_config.NumberColumn(format="%d%%"),
            "Close %": st.column_config.NumberColumn(format="%d%%"),
            "Sales $": st.column_config.NumberColumn(format="$%.0f"),
            "Avg Sale $": st.column_config.NumberColumn(format="$%.0f"),
            "Insul % of Sales": st.column_config.NumberColumn(format="%d%%"),
            "RB % of Sales":    st.column_config.NumberColumn(format="%d%%"),
            "Add-on % of Sales":st.column_config.NumberColumn(format="%d%%"),
            "Insul $": st.column_config.NumberColumn(format="$%.0f"),
            "RB $":    st.column_config.NumberColumn(format="$%.0f"),
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
                "Sales $": st.column_config.NumberColumn(format="$%.0f"),
                "Avg Sale $": st.column_config.NumberColumn(format="$%.0f"),
                "Insul $": st.column_config.NumberColumn(format="$%.0f"),
                "RB $":    st.column_config.NumberColumn(format="$%.0f"),
            }
        )

    st.subheader("Company Raw Line Items")
    mode = st.radio("Filter:", ["All", "Only Sits", "Only Sales", "Only No Shows"], horizontal=True, key="company_mode")
    comp_detail = filter_detail(detail_display, mode)
    st.dataframe(comp_detail[[c for c in DETAIL_COLS if c in comp_detail.columns]], use_container_width=True, hide_index=True)
    st.download_button("Download Company Raw CSV", data=comp_detail.to_csv(index=False).encode("utf-8"),
                       file_name="company_raw_line_items.csv", mime="text/csv")

with tab_setters:
    st.subheader("Harvester Report (Appointment Set By)")
    st.dataframe(harv_display, use_container_width=True, hide_index=True,
                 column_config={"Sit %": st.column_config.NumberColumn(format="%d%%")})

    st.markdown("### Raw Line Items for a Harvester")
    harv_list = safe_unique_sorted(harvester_report["Harvester"])
    if harv_list:
        sel_h = st.selectbox("Choose Harvester", options=harv_list, key="harv_choice")
        mode_h = st.radio("Filter:", ["All", "Only Sits", "Only Sales", "Only No Shows"], horizontal=True, key="harv_mode")
        h_detail = detail_display[ detail_display["Appointment Set By"].astype(str).str.strip() == sel_h ].copy()
        h_detail = filter_detail(h_detail, mode_h)
        st.dataframe(h_detail[[c for c in DETAIL_COLS if c in h_detail.columns]], use_container_width=True, hide_index=True)
        st.download_button(f"Download Raw CSV — {sel_h}",
                           data=h_detail.to_csv(index=False).encode("utf-8"),
                           file_name=f"harvester_raw_{sel_h.replace(' ','_')}.csv", mime="text/csv")
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
            "Avg Sale $": st.column_config.NumberColumn(format="$%.0f"),
            "Insul $": st.column_config.NumberColumn(format="$%.0f"),
            "RB $":    st.column_config.NumberColumn(format="$%.0f"),
        }
    )

    st.markdown("### Raw Line Items for a Sales Rep")
    rep_list = safe_unique_sorted(sales_report["Sales Rep"])
    if rep_list:
        sel_r = st.selectbox("Choose Sales Rep", options=rep_list, key="rep_choice")
        mode_r = st.radio("Filter:", ["All", "Only Sits", "Only Sales", "Only No Shows"], horizontal=True, key="rep_mode")
        r_detail = detail_display[ detail_display["Closer (derived)"].astype(str).str.strip() == sel_r ].copy()
        r_detail = filter_detail(r_detail, mode_r)
        st.dataframe(r_detail[[c for c in DETAIL_COLS if c in r_detail.columns]], use_container_width=True, hide_index=True)
        st.download_button(f"Download Raw CSV — {sel_r}",
                           data=r_detail.to_csv(index=False).encode("utf-8"),
                           file_name=f"sales_raw_{sel_r.replace(' ','_')}.csv", mime="text/csv")
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
        st.download_button("Download Company Total CSV", data=company_total_table.to_csv(index=False).encode("utf-8"),
                           file_name="company_total.csv", mime="text/csv")
    with colB:
        all_bytes = to_excel_bytes(
            CompanyTotal=company_total_table,
            Harvester=harv_display,
            Sales=sales_display,
            Detail=detail_display
        )
        st.download_button("Download Excel (All Sheets)", data=all_bytes,
                           file_name="weekly_reports.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.write("")
if st.button("Print Page"):
    components.html("<script>window.print();</script>", height=0)
