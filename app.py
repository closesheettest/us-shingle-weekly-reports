# app.py — US Shingle Weekly Reports (with persistent Status Rules mapping)

from __future__ import annotations
import json, hashlib, os, tempfile
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="US Shingle Weekly Reports", layout="wide", initial_sidebar_state="expanded")

# ---------- Persistence helpers for Status Rules / Mapping ----------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATUS_RULES_PATH = DATA_DIR / "status_rules.json"

def _dict_checksum(d: dict) -> str:
    j = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(j.encode("utf-8")).hexdigest()

def load_status_rules() -> dict:
    if "status_rules" in st.session_state:
        return st.session_state["status_rules"]
    if STATUS_RULES_PATH.exists():
        try:
            with STATUS_RULES_PATH.open("r", encoding="utf-8") as f:
                rules = json.load(f)
        except Exception:
            rules = {}
    else:
        rules = {}
    st.session_state["status_rules"] = rules
    st.session_state["status_rules_checksum"] = _dict_checksum(rules)
    return rules

def atomic_save_json(obj: dict, dest_path: Path):
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
        json.dump(obj, tmp, indent=2, ensure_ascii=False)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, dest_path)

def save_status_rules(rules: dict) -> bool:
    new_ck = _dict_checksum(rules)
    old_ck = st.session_state.get("status_rules_checksum")
    if new_ck == old_ck:
        return False
    atomic_save_json(rules, STATUS_RULES_PATH)
    st.session_state["status_rules"] = rules
    st.session_state["status_rules_checksum"] = new_ck
    return True

# ---------- Example Status Rules Editor UI ----------
st.title("US Shingle Weekly Reports")

rules = load_status_rules()
# Example: populate from your data (replace with real df statuses)
current_statuses = ["Sale", "Sit", "No Show", "Credit Denial", "Other"]
for s in current_statuses:
    rules.setdefault(s, {"class": "Other"})

st.subheader("Status Rules Editor")
changed = False

CLASSES = ["Sale", "Sit", "Sit-Pending", "No Show", "No Sit", "Credit Denial", "Other"]

for s in current_statuses:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**{s}**")
    with col2:
        sel = st.selectbox(
            "Class",
            CLASSES,
            index=CLASSES.index(rules[s].get("class", "Other")) if rules[s].get("class", "Other") in CLASSES else CLASSES.index("Other"),
            key=f"rule_class::{s}",
            label_visibility="collapsed",
        )
        if sel != rules[s].get("class"):
            rules[s]["class"] = sel
            changed = True

save_now = st.button("Save Status Rules")

if changed or save_now:
    if save_status_rules(rules):
        st.toast("Status rules saved ✅", icon="✅")
    else:
        st.toast("No changes to save", icon="📝")

st.caption(f"Rules file: `{STATUS_RULES_PATH}`")
