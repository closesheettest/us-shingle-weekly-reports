import streamlit as st
import json
from pathlib import Path
import pandas as pd
from report_utils import load_config_from_path, read_input_file, build_workbook

st.set_page_config(page_title="US Shingle Weekly Reports", page_icon="📊", layout="centered")

APP_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = APP_DIR / "weekly_report_config.json"
SETTINGS_PATH = APP_DIR / "settings.json"

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
    return cfg

tabs = st.tabs(["📂 Reports", "⚙️ Settings"])

with tabs[1]:
    st.header("Settings")
    st.caption("Define what counts as a Sit, Sale, and No Show. Comma-separated. Saved for future use.")
    current = get_settings()
    sit = st.text_input("Sit Statuses", value=", ".join(current.get("sit_statuses", [])))
    sale = st.text_input("Sale Statuses", value=", ".join(current.get("sale_statuses", [])))
    noshow = st.text_input("No Show Statuses", value=", ".join(current.get("no_show_statuses", [])))
    if st.button("Save Settings"):
        new_settings = save_settings(sit, sale, noshow)
        st.success("Saved! These will be used for all future reports.")
        st.json(new_settings)

with tabs[0]:
    st.title("📊 US Shingle Weekly Reports")
    st.caption("Upload your weekly CSV/Excel. The report will use your saved Settings for Sit/Sale/No-Show.")

    uploaded_data = st.file_uploader("Upload weekly data file (.csv, .xlsx, .xlsm, .xls, .xlsb)", type=["csv","xlsx","xlsm","xls","xlsb"])

    if uploaded_data is not None:
        try:
            from report_utils import load_config_from_path
            cfg = load_config_from_path(DEFAULT_CONFIG_PATH)
            cfg = merge_settings_into_config(cfg, get_settings())

            df = read_input_file(uploaded_data)
            wb_bytes = build_workbook(df, cfg)

            st.success("Report generated successfully!")
            st.download_button(
                label="⬇️ Download Weekly_Reports.xlsx",
                data=wb_bytes,
                file_name="Weekly_Reports.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            with st.expander("Preview your data (first 100 rows)"):
                st.dataframe(df.head(100))

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Upload a file on the Reports tab to generate your Excel.")
