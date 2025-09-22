from __future__ import annotations
import json, hashlib, os, tempfile
from pathlib import Path
from typing import Dict, Any

def _choose_writable_dir(app_dir: Path) -> Path:
    env_dir = os.getenv("US_SHINGLE_DATA_DIR")
    if env_dir:
        p = Path(env_dir); p.mkdir(parents=True, exist_ok=True)
        if os.access(p, os.W_OK): return p
    cloud = Path("/mount/data/us_shingle_weekly_reports")
    try:
        cloud.mkdir(parents=True, exist_ok=True)
        if os.access(cloud, os.W_OK): return cloud
    except Exception:
        pass
    local = app_dir / "data"; local.mkdir(parents=True, exist_ok=True); return local

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = _choose_writable_dir(APP_DIR)

STATUS_RULES_PATH = DATA_DIR / "status_rules.json"
LEGACY_PATH = (APP_DIR / "data" / "status_rules.json")

def _ck(d: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def _atomic_write(obj: Dict[str, Any], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=dest.parent, encoding="utf-8") as tmp:
        json.dump(obj, tmp, indent=2, ensure_ascii=False)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, dest)

def load_status_rules() -> Dict[str, Any]:
    try:
        if not STATUS_RULES_PATH.exists() and LEGACY_PATH.exists():
            STATUS_RULES_PATH.write_text(LEGACY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass
    try:
        if STATUS_RULES_PATH.exists():
            rules = json.loads(STATUS_RULES_PATH.read_text(encoding="utf-8"))
        else:
            rules = {}
    except Exception:
        rules = {}
    globals()["_last_ck"] = _ck(rules)
    return rules

def save_status_rules(rules: Dict[str, Any]) -> bool:
    new = _ck(rules); old = globals().get("_last_ck")
    if new == old: return False
    _atomic_write(rules, STATUS_RULES_PATH)
    globals()["_last_ck"] = new
    return True
