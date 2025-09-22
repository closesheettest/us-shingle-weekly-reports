US Shingle Weekly Reports — 'Sales miscount fix' look restored + mapping persistence

PAGES
- Mapping Editor
- Harvester Results
- Sales Results by Rep (with Gross & Net Closing %)
- Neal's Pay

FILES
- app.py
- status_persist.py
- data/status_rules.json  (starts empty {})

RUN
    streamlit run app.py

NOTES
- Use Mapping Editor to scan/add statuses from the chosen Status column and set checkboxes.
- Mapping is saved to data/status_rules.json (or /mount/data/... on Streamlit Cloud).
- Sales Results page shows Gross % (Sales / Sales Sits) and configurable Net %.
