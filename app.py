# app.py — US Shingle Weekly Reports (original logic preserved, mapping now persisted)

from __future__ import annotations
import streamlit as st
import pandas as pd
from status_persist import load_status_rules, save_status_rules, STATUS_RULES_PATH

st.set_page_config(page_title="US Shingle Weekly Reports", layout="wide", initial_sidebar_state="expanded")

st.title("US Shingle Weekly Reports")

# Load mapping (persisted)
rules = load_status_rules()

# Example UI placeholder for your mapping editor ------------------------------
# Replace this with your existing Status Rules editor loop
st.subheader("Status Rules Editor")

# Example statuses (replace with your df["Status"].unique())
current_statuses = ["Sale", "Sit", "No Show", "Credit Denial", "Other"]
for s in current_statuses:
    rules.setdefault(s, {"class": "Other"})

CLASSES = ["Sale", "Sit", "Sit-Pending", "No Show", "No Sit", "Credit Denial", "Other"]

changed = False
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

# -----------------------------------------------------------------------------
# All your other app code remains unchanged below this point
# (reports, metrics, etc.)
