import streamlit as st
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

from report_utils import (
    load_config_from_path, read_input_file,
    normalize_columns, ensure_columns, tag_statuses,
    compute_sales_rep_report, compute_company_totals,
    compute_harvester_report, compute_harvester_pay,
)

st.set_page_config(page_title="US Shingle Weekly Reports", page_icon="📊", layout="wide")

APP_DIR = Path(__file__).parent
CONFIG_PATH = APP_DIR / "weekly_report_config.json"
SETTINGS_PATH = APP_DIR / "settings.json"

# ---------------- Settings helpers ----------------
def get_settings():
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text())
        except Exception:
            pass
    cfg = load_config_from_path(CONFIG_PATH)
    return {
        "sit_statuses": cfg.get("sit_statuses", []),
        "sale_statuses": cfg.get("sale_statuses", []),
        "no_show_statuses": cfg.get("no_show_statuses", []),
        "exclude_statuses": [],
        "status_mapping": {}
    }

def save_settings(data: dict):
    SETTINGS_PATH.write_text(json.dumps(data, indent=2))

def merge_settings_into_config(cfg, settings):
    for k in ["sit_statuses","sale_statuses","no_show_statuses"]:
        cfg[k] = settings.get(k, cfg.get(k, []))
    return cfg

def fmt_pct_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if isinstance(c, str) and c.endswith("%"):
            out[c] = (pd.to_numeric(out[c], errors="coerce").fillna(0)*100).round(0).astype("Int64").astype(str) + "%"
    return out

def build_print_html(company, reps, harvesters, pay) -> bytes:
    style = """
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; }
      h1 { margin: 0 0 8px 0; }
      h2 { margin: 24px 0 8px 0; }
      .meta { color:#555; margin-bottom: 16px; }
      table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }
      th, td { border: 1px solid #ddd; padding: 8px; }
      th { background: #f6f6f6; text-align: left; }
    </style>
    """
    def df_html(d): return d.to_html(index=False, border=0)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = [
        style,
        "<h1>US Shingle Weekly Report</h1>",
        f"<div class='meta'>Generated: {now}</div>",
        "<h2>Company Totals</h2>", df_html(company),
        "<h2>Sales Rep Summary</h2>", df_html(reps),
        "<h2>Harvester Summary</h2>", df_html(harvesters),
        "<h2>Harvester Pay</h2>", df_html(pay)
    ]
    return "\\n".join(html).encode("utf-8")

# -------- Single uploader (shared by both tabs) --------
st.title("📊 US Shingle Weekly Reports")
st.caption("Credit Denial is always excluded from NET (Sales & Net Sit) and always counts as a Sit for Harvesters.")

uploaded = st.file_uploader("Upload one file (.csv, .xlsx, .xlsm, .xlsb) — used for **both** Mapping and Reports",
                            type=["csv","xlsx","xlsm","xlsb"])

@st.cache_data(show_spinner=False)
def _load_df(file_bytes: bytes, file_name: str):
    from io import BytesIO
    class Up:
        def __init__(self, name, data): self.name, self._b = name, BytesIO(data)
        def read(self, *a, **k): return self._b.read(*a, **k)
        def seek(self, *a, **k): return self._b.seek(*a, **k)
    cfg = load_config_from_path(CONFIG_PATH)
    up = Up(file_name, file_bytes)
    df = read_input_file(up)
    df = normalize_columns(df)
    cols_map = ensure_columns(df, cfg)
    return df, cols_map, cfg

df_shared = None
cols_shared = None
cfg_base = None
if uploaded is not None:
    try:
        file_bytes = uploaded.getvalue()
        df_shared, cols_shared, cfg_base = _load_df(file_bytes, uploaded.name)
        st.success(f"Loaded: {uploaded.name}  •  {len(df_shared):,} rows")
    except Exception as e:
        st.error(f"Could not read file: {e}")

tabs = st.tabs(["🔧 Settings", "📂 Reports"])

# ---------------- Settings Tab ----------------
with tabs[0]:
    st.subheader("Defaults (used if a per-status mapping is not set)")
    s = get_settings()

    c1, c2 = st.columns(2)
    with c1:
        sit = st.text_input("Default Sit Statuses (comma-separated)", value=", ".join(s.get("sit_statuses", [])))
        noshow = st.text_input("Default No Show Statuses (comma-separated)", value=", ".join(s.get("no_show_statuses", [])))
    with c2:
        sale = st.text_input("Default Sale Statuses (comma-separated)", value=", ".join(s.get("sale_statuses", [])))
        st.caption("Exclude (NET) is hard-coded to 'Credit Denial'.")

    if st.button("Save Default Lists"):
        s["sit_statuses"] = [x.strip() for x in sit.split(",") if x.strip()]
        s["sale_statuses"] = [x.strip() for x in sale.split(",") if x.strip()]
        s["no_show_statuses"] = [x.strip() for x in noshow.split(",") if x.strip()]
        SETTINGS_PATH.write_text(json.dumps(s, indent=2))
        st.success("Default lists saved.")

    st.divider()
    st.subheader("Status Mapping (per status)")
    st.caption("Uses the same uploaded file to detect Status values. Check boxes and click **Save Mapping**. "
               "Credit Denial is always counted as Sit (Harvester) and excluded from NET.")

    if df_shared is None:
        st.info("Upload a file above to configure mapping.")
    else:
        col_status = cols_shared["status"]
        statuses = sorted(df_shared[col_status].dropna().astype(str).str.strip().unique().tolist())
        st.write(f"Found **{len(statuses)}** unique statuses.")

        current = s.get("status_mapping", {})
        with st.form("status_mapping_form", clear_on_submit=False):
            st.write("Tick the boxes for each status:")
            hdr = st.columns([3,1.2,1.2,1.6,1.6])
            hdr[0].markdown("**Status**")
            hdr[1].markdown("**Sale (Sales)**")
            hdr[2].markdown("**Sit (Sales)**")
            hdr[3].markdown("**Sit (Harvester)**")
            hdr[4].markdown("**No Sit (Sales)**")
            extra = st.columns([1.6])
            extra[0].markdown("**No Sit (Harvester)**")

            new_map = {}
            for st_name in statuses:
                row = st.columns([3,1.2,1.2,1.6,1.6,1.6])
                flags = current.get(st_name, {})
                is_cd = st_name.lower().strip() == "credit denial"

                if is_cd:
                    sale_b = True
                    sit_s  = True
                    sit_h  = True
                    ns_s   = False
                    ns_h   = False
                    row[1].checkbox("", value=True, key=f"sale_{st_name}", disabled=True)
                    row[2].checkbox("", value=True, key=f"sitS_{st_name}", disabled=True)
                    row[3].checkbox("", value=True, key=f"sitH_{st_name}", disabled=True)
                    row[4].checkbox("", value=False, key=f"nsS_{st_name}", disabled=True)
                    row[5].checkbox("", value=False, key=f"nsH_{st_name}", disabled=True)
                else:
                    sale_b = row[1].checkbox("", value=bool(flags.get("is_sale")), key=f"sale_{st_name}")
                    sit_s  = row[2].checkbox("", value=bool(flags.get("is_sit_sales")), key=f"sitS_{st_name}")
                    sit_h  = row[3].checkbox("", value=bool(flags.get("is_sit_harvester")), key=f"sitH_{st_name}")
                    ns_s   = row[4].checkbox("", value=bool(flags.get("is_no_sit_sales")), key=f"nsS_{st_name}")
                    ns_h   = row[5].checkbox("", value=bool(flags.get("is_no_sit_harvester")), key=f"nsH_{st_name}")

                row[0].write(st_name)
                new_map[st_name] = {
                    "is_sale": sale_b,
                    "is_sit_sales": sit_s,
                    "is_sit_harvester": sit_h,
                    "is_no_sit_sales": ns_s,
                    "is_no_sit_harvester": ns_h
                }

            if st.form_submit_button("Save Mapping"):
                s["status_mapping"] = new_map
                SETTINGS_PATH.write_text(json.dumps(s, indent=2))
                st.success("Status mapping saved.")

# ---------------- Reports Tab ----------------
with tabs[1]:
    st.subheader("Reports")
    if df_shared is None:
        st.info("Upload a file above to see reports.")
    else:
        settings = get_settings()
        cfg = merge_settings_into_config(cfg_base, settings)

        tagged = tag_statuses(df_shared, cols_shared["status"], cfg, settings).copy()
        tagged["Harvester"] = (
            tagged[cols_shared["appointment_set_by"]].fillna("").astype(str).str.strip().replace("", "Company")
        )

        # Build summaries
        rep  = compute_sales_rep_report(tagged, cols_shared)
        comp = compute_company_totals(tagged, cols_shared)
        harv = compute_harvester_report(tagged, cols_shared)
        pay  = compute_harvester_pay(harv)

        # Show
        t1, t2, t3, t4, t5 = st.tabs([
            "Sales Rep Summary", "Company Totals", "Harvester Summary", "Harvester Pay", "Print / Export"
        ])

        with t1:
            desired_order = [
                "Sales Rep","Appointments","Sits","Sit %","Net Sit","Net Sit %","Sales","Sales %",
                "Net Sales","Net Sales %","insulation (dollar amount)","radiant Barrier (dollar Amount)","irbad %"
            ]
            rep = rep[[c for c in desired_order if c in rep.columns]]
            st.dataframe(fmt_pct_cols(rep), use_container_width=True)

        with t2:
            st.dataframe(fmt_pct_cols(comp), use_container_width=True)

        with t3:
            st.dataframe(harv, use_container_width=True)

        with t4:
            st.dataframe(pay, use_container_width=True)

        with t5:
            # Print-ready HTML
            html_bytes = build_print_html(
                fmt_pct_cols(comp), fmt_pct_cols(rep), harv, pay
            )
            st.download_button(
                "⬇️ Download Print-Ready HTML",
                data=html_bytes,
                file_name="US_Shingle_Weekly_Report_Print.html",
                mime="text/html",
                help="Open in your browser, then File → Print (or Save as PDF)."
            )

            # Excel workbook (built here so it uses current mapping)
            import io
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
                rep.to_excel(xw, sheet_name="Sales Rep Summary", index=False)
                comp.to_excel(xw, sheet_name="Company Totals", index=False)
                harv.to_excel(xw, sheet_name="Harvester Summary", index=False)
                pay.to_excel(xw, sheet_name="Harvester Pay", index=False)
            st.download_button(
                "⬇️ Download Weekly_Reports.xlsx",
                data=bio.getvalue(),
                file_name="Weekly_Reports.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
